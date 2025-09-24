from __future__ import annotations

from typing import Optional

from ..settings import Settings
from .base import SearchClient
from .brave import BraveSearchClient
from .serpapi import SerpAPISearchClient
from ..api_config import ApiConfig, load_api_config


def get_search_client(settings: Settings, api_config: ApiConfig | None = None) -> SearchClient:
    cfg = api_config or load_api_config(settings.api_config_path)
    if settings.search_provider == "serpapi":
        api_key = settings.serpapi_api_key or (cfg.serpapi.api_key if cfg else None)
        if not api_key:
            raise RuntimeError(
                "SerpAPI key missing. Set SERPAPI_API_KEY, put it in apis.yaml, or pass --serpapi-key."
            )
        return SerpAPISearchClient(
            api_key=api_key,
            user_agent=settings.user_agent,
            request_timeout=settings.request_timeout,
        )
    if settings.search_provider == "brave":
        api_key = settings.brave_api_key or (cfg.brave.api_key if cfg else None)
        if not api_key:
            raise RuntimeError(
                "Brave key missing. Set BRAVE_API_KEY or put it in apis.yaml."
            )
        return BraveSearchClient(
            api_key=api_key,
            user_agent=settings.user_agent,
            request_timeout=settings.request_timeout,
        )
    raise ValueError(f"Unsupported search provider: {settings.search_provider}")
