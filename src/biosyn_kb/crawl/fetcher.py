from __future__ import annotations

import asyncio
import time
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import httpx

from .models import Page
from .robots import RobotsCache


class _PerDomainLimiter:
    def __init__(self, delay_seconds: float = 1.0):
        self._delay = delay_seconds
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_at: Dict[str, float] = {}

    async def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            last = self._last_at.get(host, 0.0)
            to_wait = self._delay - (now - last)
            if to_wait > 0:
                await asyncio.sleep(to_wait)
            self._last_at[host] = time.monotonic()


class Crawler:
    def __init__(
        self,
        user_agent: str = "biosyn-kb-agent/0.1",
        request_timeout: float = 15.0,
        per_domain_delay: float = 1.0,
        max_concurrency: int = 8,
    ):
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
            timeout=request_timeout,
            follow_redirects=True,
        )
        self._robots = RobotsCache(user_agent=user_agent, client=self._client)
        self._limiter = _PerDomainLimiter(delay_seconds=per_domain_delay)
        self._sem = asyncio.Semaphore(max_concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "Crawler":  # type: ignore[override]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:  # type: ignore[override]
        await self.aclose()
        return None

    async def fetch_one(self, url: str) -> Page:
        allowed = await self._robots.allowed(url)
        if not allowed:
            return Page(url=url, final_url=url, status=0, content_type=None, encoding=None, html=None, error="Disallowed by robots.txt")
        await self._limiter.wait(url)
        async with self._sem:
            try:
                res = await self._client.get(url)
                ct = res.headers.get("content-type", "")
                html = None
                if "text/html" in ct or "application/xhtml+xml" in ct:
                    html = res.text
                return Page(
                    url=url,
                    final_url=str(res.request.url),
                    status=res.status_code,
                    content_type=ct,
                    encoding=res.encoding,
                    html=html,
                )
            except Exception as e:
                return Page(url=url, final_url=url, status=0, content_type=None, encoding=None, html=None, error=str(e))

    async def fetch_many(self, urls: Iterable[str]) -> List[Page]:
        tasks = [self.fetch_one(u) for u in urls]
        return await asyncio.gather(*tasks)

