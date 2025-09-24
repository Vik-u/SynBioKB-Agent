from __future__ import annotations

from typing import List, Optional

import httpx

from .base import SearchResult


class SerpAPISearchClient:
    """SerpAPI client for Google Web results.

    Docs: https://serpapi.com/search-api
    """

    BASE_URL = "https://serpapi.com/search.json"

    def __init__(self, api_key: str, user_agent: str, request_timeout: float = 15.0):
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=request_timeout,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "SerpAPISearchClient":  # type: ignore[override]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        await self.aclose()
        return None

    async def search(self, query: str, count: int = 10, offset: int = 0) -> List[SearchResult]:
        # SerpAPI (Google) supports num up to 10 per page, use start for offset
        num = max(1, min(10, count))
        params = {
            "engine": "google",
            "q": query,
            "num": num,
            "start": offset,
            "api_key": self._api_key,
        }
        resp = await self._client.get(self.BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("organic_results") or []
        results: List[SearchResult] = []
        for item in items:
            url = item.get("link") or ""
            title = item.get("title") or ""
            snippet = item.get("snippet")
            rank = item.get("position")
            results.append(
                SearchResult(title=title, url=url, snippet=snippet, rank=rank, source="serpapi")
            )
        return results

