from __future__ import annotations

import asyncio
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import httpx
from playwright.async_api import async_playwright

from .models import Page
from .robots import RobotsCache


class _PerDomainLimiter:
    def __init__(self, delay_seconds: float = 1.0):
        self._delay = delay_seconds
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_at: Dict[str, float] = {}

    async def wait(self, url: str) -> None:
        import time

        host = urlparse(url).netloc
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            last = self._last_at.get(host, 0.0)
            to_wait = self._delay - (now - last)
            if to_wait > 0:
                await asyncio.sleep(to_wait)
            self._last_at[host] = time.monotonic()


class HeadlessCrawler:
    """Headless (JS-rendered) crawler using Playwright Chromium.

    Respects robots.txt via RobotsCache, has per-domain throttling, and limits concurrency.
    Only used when explicitly requested (heavier dependency footprint).
    """

    def __init__(
        self,
        user_agent: str = "biosyn-kb-agent/0.1",
        request_timeout: float = 20.0,
        per_domain_delay: float = 1.0,
        max_concurrency: int = 4,
    ):
        self._ua = user_agent
        self._timeout = request_timeout
        self._limiter = _PerDomainLimiter(delay_seconds=per_domain_delay)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._http = httpx.AsyncClient(timeout=10.0)
        self._robots = RobotsCache(user_agent=user_agent, client=self._http)
        self._pw = None
        self._browser = None

    async def __aenter__(self) -> "HeadlessCrawler":
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=True)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:
        await self.aclose()
        return None

    async def aclose(self) -> None:
        try:
            if self._browser:
                await self._browser.close()
        finally:
            self._browser = None
        try:
            if self._pw:
                await self._pw.stop()
        finally:
            self._pw = None
        await self._http.aclose()

    async def fetch_one(self, url: str) -> Page:
        allowed = await self._robots.allowed(url)
        if not allowed:
            return Page(url=url, final_url=url, status=0, content_type=None, encoding=None, html=None, error="Disallowed by robots.txt")
        await self._limiter.wait(url)
        async with self._sem:
            ctx = await self._browser.new_context(user_agent=self._ua)
            page = await ctx.new_page()
            try:
                resp = await page.goto(url, timeout=int(self._timeout * 1000), wait_until="domcontentloaded")
                content = await page.content()
                final_url = page.url
                status = resp.status if resp else 0
                ct = None
                try:
                    ct = resp.headers.get("content-type") if resp else None
                except Exception:
                    ct = None
                return Page(
                    url=url,
                    final_url=final_url,
                    status=status,
                    content_type=ct,
                    encoding="utf-8",
                    html=content,
                )
            except Exception as e:
                return Page(url=url, final_url=url, status=0, content_type=None, encoding=None, html=None, error=str(e))
            finally:
                try:
                    await page.close()
                except Exception:
                    pass
                try:
                    await ctx.close()
                except Exception:
                    pass

    async def fetch_many(self, urls: Iterable[str]) -> List[Page]:
        tasks = [self.fetch_one(u) for u in urls]
        return await asyncio.gather(*tasks)

