from __future__ import annotations

from typing import Optional

from .ollama_client import OllamaClient
from .openai_client import OpenAIClient


def get_llm_client(provider: str = "ollama", base_url: str = "http://localhost:11434", api_key: Optional[str] = None):
    p = (provider or "ollama").lower()
    if p == "ollama":
        return OllamaClient(base_url)
    if p in ("openai", "openai-compatible"):
        return OpenAIClient(base_url=base_url or "https://api.openai.com/v1", api_key=api_key)
    # Default back to Ollama if unknown
    return OllamaClient(base_url)

