"""OpenAI client configuration."""
from openai import OpenAI
from config.settings import settings


_client = None


def get_openai_client() -> OpenAI:
    """Create the OpenAI client only when an OpenAI-backed feature is used."""
    global _client
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for this operation")
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


