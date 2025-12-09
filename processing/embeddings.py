"""Embedding generation using OpenAI or Sentence Transformers."""
from typing import List
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Lazy loading of embedding models
_sentence_transformer_model = None
_openai_client = None


def _get_sentence_transformer():
    """Lazy load sentence transformer model."""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence transformer model: {settings.sentence_transformer_model}")
            _sentence_transformer_model = SentenceTransformer(settings.sentence_transformer_model)
            logger.info("Sentence transformer model loaded successfully")
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
    return _sentence_transformer_model


def _get_openai_client():
    """Lazy load OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from config.openai_client import client
        _openai_client = client
    return _openai_client


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text.
    
    Uses OpenAI or Sentence Transformers based on settings.
    
    Args:
        text: Input text to embed
        
    Returns:
        Embedding vector as list of floats
    """
    if settings.embedding_provider == "sentence-transformers":
        # Use free sentence transformers
        model = _get_sentence_transformer()
        embedding = model.encode(text, convert_to_numpy=False)
        logger.debug(f"Generated embedding of dimension {len(embedding)} using sentence-transformers")
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
    
    elif settings.embedding_provider == "openai":
        # Use OpenAI
        client = _get_openai_client()
        try:
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=text
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding of dimension {len(embedding)} using OpenAI")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding with OpenAI: {e}")
            raise ValueError(f"Failed to generate embedding: {e}")
    
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def generate_embeddings_batch(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to process per batch
        
    Returns:
        List of embedding vectors
    """
    if settings.embedding_provider == "sentence-transformers":
        # Sentence transformers handles batching efficiently
        model = _get_sentence_transformer()
        logger.info(f"Generating embeddings for {len(texts)} texts using sentence-transformers")
        embeddings = model.encode(texts, convert_to_numpy=True, batch_size=batch_size)
        # Convert numpy array to list of lists
        import numpy as np
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        # Ensure it's a list of lists
        if isinstance(embeddings, list) and len(embeddings) > 0:
            if not isinstance(embeddings[0], list):
                embeddings = [embeddings]  # Single embedding case
        logger.info(f"Generated {len(embeddings)} embeddings, each of dimension {len(embeddings[0]) if embeddings else 0}")
        return embeddings
    
    elif settings.embedding_provider == "openai":
        # OpenAI batching
        client = _get_openai_client()
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {i // batch_size + 1} ({len(batch)} texts)")
            
            try:
                response = client.embeddings.create(
                    model=settings.embedding_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                raise ValueError(f"Failed to generate embeddings for batch: {e}")
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
