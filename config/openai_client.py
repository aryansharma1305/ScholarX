"""OpenAI client configuration."""
from openai import OpenAI
from config.settings import settings

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)



