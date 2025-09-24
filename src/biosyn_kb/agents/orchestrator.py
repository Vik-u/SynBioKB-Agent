from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..extract.clean import extract_clean_text
from ..summarize.pipeline import summarize_pages_file
from ..store import init_db, import_pages_from_html_dir, import_summaries_jsonl


@dataclass
class PipelineConfig:
    html_dir: Path
    work_dir: Path
    db_path: Path
    summaries_jsonl: Path
    model: str = "gpt-oss:20b"
    provider: str = "ollama"
    ollama_url: str = "http://localhost:11434"  # kept for backward compat
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    limit: Optional[int] = None
    max_chars: int = 12000
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    skip_llm: bool = False


async def run_local_pipeline(cfg: PipelineConfig) -> None:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    # 1) Extract & store full pages in DB
    init_db(cfg.db_path)
    count = import_pages_from_html_dir(cfg.db_path, cfg.html_dir, use_cleaner=True, include_pdfs=True)
    # 2) Build extracted JSONL from DB-imported HTML (re-use cleaner again into a JSONL for LLM)
    extracted_jsonl = cfg.work_dir / "extracted.jsonl"
    import json
    def _escape(s: str) -> str:
        return s.replace("\\", " ").replace("\n", " ")
    with extracted_jsonl.open("w", encoding="utf-8") as sink:
        for p in sorted(cfg.html_dir.glob("*.html")):
            html = p.read_text(encoding="utf-8", errors="ignore")
            clean = extract_clean_text(html)
            rec = {"file": str(p), "title": (clean.title or ""), "text": _escape(clean.text or "")}
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # 3) Summarize into structured records
    if not cfg.skip_llm:
        await summarize_pages_file(
            input_jsonl=extracted_jsonl,
            output_jsonl=cfg.summaries_jsonl,
            model=cfg.model,
            provider=cfg.provider,
            base_url=(cfg.base_url or cfg.ollama_url),
            api_key=cfg.api_key,
            limit=cfg.limit,
            concurrency=1,
            max_chars=cfg.max_chars,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
        )
        # 4) Import summaries into DB
        import_summaries_jsonl(cfg.db_path, cfg.summaries_jsonl)
