from __future__ import annotations

import asyncio
from typing import List, Optional

import httpx

from .base import SearchResult


class BraveSearchClient:
    """Brave Search API client (Web results).

    Docs: https://api.search.brave.com/app/documentation/web-search
    Requires env header X-Subscription-Token with the API key.
    """

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, user_agent: str, request_timeout: float = 15.0):
        headers = {
            "User-Agent": user_agent,
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        self._client = httpx.AsyncClient(timeout=request_timeout, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BraveSearchClient":  # type: ignore[override]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        await self.aclose()
        return None

    async def search(self, query: str, count: int = 10, offset: int = 0) -> List[SearchResult]:
        params = {
            "q": query,
            "count": count,
            "offset": offset,
            # You can tune these if needed
            "safesearch": "moderate",
        }
        resp = await self._client.get(self.BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        results: List[SearchResult] = []
        web = data.get("web") or {}
        items = web.get("results") or []
        for idx, item in enumerate(items, start=1 + offset):
            url = item.get("url") or item.get("page_url") or ""
            title = item.get("title") or ""
            snippet = item.get("description") or item.get("snippet")
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    rank=idx,
                    source="brave",
                )
            )
        return results


async def _demo():
    import os

    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        print("Missing BRAVE_API_KEY; set it to run the demo.")
        return
    query = "biosynthesis of caffeine"
    async with BraveSearchClient(api_key, user_agent="biosyn-kb-agent/0.1") as client:
        res = await client.search(query, count=5)
        for r in res:
            print(r)


if __name__ == "__main__":
    asyncio.run(_demo())

