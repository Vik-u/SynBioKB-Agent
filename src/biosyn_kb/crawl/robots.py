from __future__ import annotations

import asyncio
from typing import Dict
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


class RobotsCache:
    def __init__(self, user_agent: str, client: httpx.AsyncClient):
        self._user_agent = user_agent
        self._client = client
        self._cache: Dict[str, RobotFileParser] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        rp = await self._get_rp(base)
        return rp.can_fetch(self._user_agent, url)

    async def _get_rp(self, base: str) -> RobotFileParser:
        if base in self._cache:
            return self._cache[base]
        lock = self._locks.setdefault(base, asyncio.Lock())
        async with lock:
            if base in self._cache:
                return self._cache[base]
            rp = RobotFileParser()
            robots_url = f"{base}/robots.txt"
            try:
                res = await self._client.get(robots_url, timeout=10)
                if res.status_code == 200 and res.text:
                    rp.parse(res.text.splitlines())
                else:
                    rp.parse([])
            except Exception:
                rp.parse([])
            self._cache[base] = rp
            return rp

