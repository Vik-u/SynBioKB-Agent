from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str | None = None
    rank: int | None = None
    source: str | None = None


class SearchClient(Protocol):
    async def search(self, query: str, count: int = 10, offset: int = 0) -> List[SearchResult]:
        ...

