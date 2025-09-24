from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import httpx
from bs4 import BeautifulSoup


def find_pdf_links_in_html(html: str, base_url: str | None = None) -> List[str]:
    urls: List[str] = []
    soup = BeautifulSoup(html, "html.parser")
    # citation_pdf_url meta
    for m in soup.find_all("meta", attrs={"name": "citation_pdf_url"}):
        href = m.get("content")
        if href and href.lower().endswith(".pdf"):
            urls.append(href)
    # link rel alternate
    for l in soup.find_all("link"):
        if (l.get("type") == "application/pdf") or (l.get("rel") and "alternate" in l.get("rel", [])):
            href = l.get("href")
            if href and href.lower().endswith(".pdf"):
                urls.append(href)
    # anchor tags
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        if href.lower().endswith(".pdf"):
            urls.append(href)
    # de-dup
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


async def download_pdfs_from_urls(urls: Iterable[str], out_dir: str | Path, *, user_agent: str = "biosyn-kb-agent/0.1") -> List[Tuple[str, str]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Tuple[str, str]] = []
    async with httpx.AsyncClient(timeout=60.0, headers={"User-Agent": user_agent}, follow_redirects=True) as client:
        for url in urls:
            try:
                r = await client.get(url)
                ct = r.headers.get("content-type", "")
                if r.status_code == 200 and ("application/pdf" in ct or url.lower().endswith(".pdf")):
                    name = _safe_name(url)
                    path = out_dir / f"{name}.pdf"
                    path.write_bytes(r.content)
                    results.append((url, str(path)))
            except Exception:
                continue
    return results


def _safe_name(url: str) -> str:
    import re
    s = url.replace("://", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]", "_", s)
    return s[:220]

