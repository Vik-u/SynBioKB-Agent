from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .text import extract_main_text as _bs4_extract


@dataclass
class CleanContent:
    title: Optional[str]
    text: str
    method: str


def extract_clean_text(html: str, url: str | None = None) -> CleanContent:
    try:
        import trafilatura

        extracted = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
        if extracted and extracted.strip():
            # Try to parse metadata for title
            meta = trafilatura.extract_metadata(html, url=url)
            title = meta.title if meta and getattr(meta, "title", None) else None
            return CleanContent(title=title, text=extracted.strip(), method="trafilatura")
    except Exception:
        pass
    # Fallback to bs4-based simple extractor
    bs = _bs4_extract(html)
    return CleanContent(title=bs.title, text=bs.text, method="bs4")

