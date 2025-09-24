from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Page:
    url: str
    final_url: str
    status: int
    content_type: Optional[str]
    encoding: Optional[str]
    html: Optional[str]
    error: Optional[str] = None

