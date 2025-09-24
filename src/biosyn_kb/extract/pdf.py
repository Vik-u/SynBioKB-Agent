from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PDFContent:
    title: Optional[str]
    text: str


def extract_pdf_text(path: str) -> PDFContent:
    # Try pdfplumber first
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            parts = []
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
            text = "\n\n".join(parts)
            return PDFContent(title=None, text=text)
    except Exception:
        pass
    # Fallback to pdfminer.six high-level
    try:
        from pdfminer.high_level import extract_text

        text = extract_text(path) or ""
        return PDFContent(title=None, text=text)
    except Exception:
        return PDFContent(title=None, text="")

