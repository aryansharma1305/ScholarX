"""Timer utilities for performance measurement."""
import time
from contextlib import contextmanager
from typing import Generator
from utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def timer(operation_name: str) -> Generator[None, None, None]:
    """
    Context manager for timing operations.
    
    Usage:
        with timer("Processing documents"):
            # Your code here
            pass
    """
    start_time = time.time()
    logger.info(f"Starting: {operation_name}")
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed: {operation_name} in {elapsed:.2f} seconds")


def time_function(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper



