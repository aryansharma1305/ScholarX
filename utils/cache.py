"""Simple caching layer for API responses and embeddings."""
from typing import Any, Optional, Dict
from functools import wraps
import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Cache TTL (time to live) in seconds
CACHE_TTL = {
    "embeddings": 86400 * 7,  # 7 days
    "api_responses": 3600,  # 1 hour
    "search_results": 1800,  # 30 minutes
    "summaries": 86400,  # 1 day
}


def get_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from function arguments."""
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items())
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cache_path(cache_type: str, key: str) -> Path:
    """Get cache file path."""
    return CACHE_DIR / f"{cache_type}_{key}.pkl"


def load_from_cache(cache_type: str, key: str) -> Optional[Any]:
    """Load data from cache if not expired."""
    cache_path = get_cache_path(cache_type, key)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Check expiration
        if "timestamp" in cached_data and "ttl" in cached_data:
            age = (datetime.now() - cached_data["timestamp"]).total_seconds()
            if age > cached_data["ttl"]:
                cache_path.unlink()  # Delete expired cache
                return None
        
        logger.debug(f"Cache hit: {cache_type}/{key}")
        return cached_data.get("data")
        
    except Exception as e:
        logger.warning(f"Error loading cache: {e}")
        return None


def save_to_cache(cache_type: str, key: str, data: Any, ttl: Optional[int] = None):
    """Save data to cache."""
    cache_path = get_cache_path(cache_type, key)
    
    if ttl is None:
        ttl = CACHE_TTL.get(cache_type, 3600)
    
    try:
        cached_data = {
            "data": data,
            "timestamp": datetime.now(),
            "ttl": ttl
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        
        logger.debug(f"Cache saved: {cache_type}/{key}")
        
    except Exception as e:
        logger.warning(f"Error saving cache: {e}")


def cached(cache_type: str = "api_responses", ttl: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache_type: Type of cache (determines default TTL)
        ttl: Time to live in seconds (overrides default)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = get_cache_key(func.__name__, *args, **kwargs)
            
            # Try to load from cache
            cached_result = load_from_cache(cache_type, cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Save to cache
            save_to_cache(cache_type, cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def clear_cache(cache_type: Optional[str] = None):
    """Clear cache files."""
    if cache_type:
        pattern = f"{cache_type}_*.pkl"
        for cache_file in CACHE_DIR.glob(pattern):
            cache_file.unlink()
        logger.info(f"Cleared {cache_type} cache")
    else:
        # Clear all
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cleared all cache")


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    
    stats = {
        "total_files": len(cache_files),
        "total_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
        "by_type": {}
    }
    
    # Group by type
    for cache_file in cache_files:
        cache_type = cache_file.stem.split("_")[0]
        stats["by_type"][cache_type] = stats["by_type"].get(cache_type, 0) + 1
    
    return stats

