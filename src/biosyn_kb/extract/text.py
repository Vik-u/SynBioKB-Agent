from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup


@dataclass
class ExtractedContent:
    title: str | None
    text: str


def extract_main_text(html: str) -> ExtractedContent:
    soup = BeautifulSoup(html, "html.parser")

    # Drop non-content elements
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()

    # Try to reduce boilerplate by removing nav/footer/aside tags if present
    for tag_name in ("nav", "footer", "aside"):
        for t in soup.find_all(tag_name):
            t.decompose()

    # Title
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Fallback: largest <p>-cluster text, else full text
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    if paragraphs:
        text = "\n\n".join([p for p in paragraphs if p])
    else:
        text = soup.get_text("\n", strip=True)

    # Normalize whitespace: collapse 3+ newlines
    import re

    text = re.sub(r"\n{3,}", "\n\n", text)
    return ExtractedContent(title=title, text=text)

