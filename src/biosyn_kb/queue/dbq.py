from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..crawl import Crawler
from ..summarize.pipeline import summarize_pages_file
from ..search.clients import get_search_client
from ..api_config import load_api_config
from ..settings import get_settings


@dataclass
class QueueDB:
    path: Path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              type TEXT NOT NULL,
              payload TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending',
              created_at REAL NOT NULL,
              started_at REAL,
              finished_at REAL,
              error TEXT
            )
            """
        )
        conn.commit()
        conn.close()


def enqueue_job(qdb: QueueDB, job_type: str, payload: Dict[str, Any]) -> int:
    qdb.init()
    conn = qdb.connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO jobs(type, payload, created_at) VALUES(?,?,?)",
        (job_type, json.dumps(payload, ensure_ascii=False), time.time()),
    )
    job_id = cur.lastrowid
    conn.commit()
    conn.close()
    return int(job_id)


async def _crawl_do(payload: Dict[str, Any]) -> None:
        urls = payload.get("urls") or []
        save_dir = Path(payload.get("save_html_dir") or "pages/queue")
        save_dir.mkdir(parents=True, exist_ok=True)
        from ..settings import get_settings

        settings = get_settings()
        async with Crawler(
            user_agent=settings.user_agent,
            request_timeout=settings.request_timeout,
            per_domain_delay=1.0,
            max_concurrency=4,
        ) as crawler:
            pages = await crawler.fetch_many(urls)
            for p in pages:
                if p.html:
                    filename = (p.final_url or p.url).replace("://", "_")
                    filename = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in filename)
                    (save_dir / f"{filename}.html").write_text(p.html, encoding=p.encoding or "utf-8", errors="ignore")


async def _summarize_do(payload: Dict[str, Any]) -> None:
        inp = Path(payload["input_jsonl"])  # required
        out = Path(payload["output_jsonl"])  # required
        model = payload.get("model") or "gpt-oss:20b"
        # Support both legacy 'ollama_url' and new provider/base_url/api_key
        provider = payload.get("provider") or "ollama"
        base_url = payload.get("base_url") or payload.get("ollama_url") or "http://localhost:11434"
        api_key = payload.get("api_key")
        limit = payload.get("limit")
        concurrency = int(payload.get("concurrency") or 1)
        max_chars = int(payload.get("max_chars") or 6000)
        temperature = float(payload.get("temperature") or 0.0)
        top_p = float(payload.get("top_p") or 1.0)
        seed = payload.get("seed")
        use_filter = bool(payload.get("use_filter") or True)
        chunked = bool(payload.get("chunked") or True)
        chunk_chars = int(payload.get("chunk_chars") or 2800)
        await summarize_pages_file(
            input_jsonl=inp,
            output_jsonl=out,
            model=model,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            limit=limit,
            concurrency=concurrency,
            max_chars=max_chars,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            use_filter=use_filter,
            chunked=chunked,
            chunk_chars=chunk_chars,
        )


def run_worker_once(qdb: QueueDB) -> Optional[int]:
    """Process one pending job if any. Returns job id processed or None if no jobs."""
    qdb.init()
    conn = qdb.connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY id ASC LIMIT 1")
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    job_id = int(row["id"])
    job_type = row["type"]
    payload = json.loads(row["payload"]) or {}
    cur.execute("UPDATE jobs SET status='in_progress', started_at=? WHERE id=?", (time.time(), job_id))
    conn.commit()
    conn.close()
    try:
        import asyncio
        if job_type == "crawl":
            asyncio.run(_crawl_do(payload))
        elif job_type == "summarize":
            asyncio.run(_summarize_do(payload))
        elif job_type == "agent":
            asyncio.run(_agent_do(payload))
        elif job_type == "crew":
            from ..agents import run_crewai_or_fallback
            asyncio.run(
                run_crewai_or_fallback(
                    html_dir=Path(payload["html_dir"]),
                    work_dir=Path(payload["work_dir"]),
                    db_path=Path(payload["db_path"]),
                    out_jsonl=Path(payload["out_jsonl"]),
                    model=payload.get("model") or "gpt-oss:20b",
                    ollama_url=payload.get("ollama_url") or "http://localhost:11434",
                    limit=payload.get("limit"),
                    max_chars=int(payload.get("max_chars") or 6000),
                    temperature=float(payload.get("temperature") or 0.0),
                    top_p=float(payload.get("top_p") or 1.0),
                    seed=int(payload.get("seed") or 42),
                )
            )
        elif job_type == "e2e":
            from ..agents.crewai_impl import run_crewai_end_to_end_async
            asyncio.run(
                run_crewai_end_to_end_async(
                    query=payload.get("query") or "",
                    work_dir=Path(payload.get("work_dir") or "work/e2e"),
                    html_dir=Path(payload.get("html_dir") or "pages/e2e"),
                    db_path=Path(payload.get("db_path") or "artifacts/e2e.db"),
                    out_jsonl=Path(payload.get("out_jsonl") or "artifacts/e2e_summaries.jsonl"),
                    api_config_path=payload.get("api_config_path"),
                    provider=payload.get("provider") or "ollama",
                    base_url=payload.get("base_url") or "http://localhost:11434",
                    api_key=payload.get("api_key"),
                    model=payload.get("model") or "gpt-oss:20b",
                    limit=int(payload.get("limit") or 5),
                    max_results=int(payload.get("max_results") or 50),
                    max_chars=int(payload.get("max_chars") or 12000),
                    temperature=float(payload.get("temperature") or 0.0),
                    top_p=float(payload.get("top_p") or 1.0),
                    seed=int(payload.get("seed") or 42),
                )
            )
        else:
            raise ValueError(f"Unknown job type: {job_type}")
        conn2 = qdb.connect()
        conn2.execute("UPDATE jobs SET status='done', finished_at=? WHERE id=?", (time.time(), job_id))
        conn2.commit()
        conn2.close()
    except Exception as e:
        conn3 = qdb.connect()
        conn3.execute(
            "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
            (time.time(), str(e), job_id),
        )
        conn3.commit()
        conn3.close()
    return job_id


async def run_worker_once_async(qdb: QueueDB) -> Optional[int]:
    qdb.init()
    conn = qdb.connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY id ASC LIMIT 1")
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    job_id = int(row["id"])
    job_type = row["type"]
    payload = json.loads(row["payload"]) or {}
    cur.execute("UPDATE jobs SET status='in_progress', started_at=? WHERE id=?", (time.time(), job_id))
    conn.commit()
    conn.close()
    try:
        if job_type == "crawl":
            await _crawl_do(payload)
        elif job_type == "summarize":
            await _summarize_do(payload)
        elif job_type == "agent":
            await _agent_do(payload)
        elif job_type == "crew":
            from ..agents import run_crewai_or_fallback
            await run_crewai_or_fallback(
                html_dir=Path(payload["html_dir"]),
                work_dir=Path(payload["work_dir"]),
                db_path=Path(payload["db_path"]),
                out_jsonl=Path(payload["out_jsonl"]),
                model=payload.get("model") or "gpt-oss:20b",
                ollama_url=payload.get("ollama_url") or "http://localhost:11434",
                limit=payload.get("limit"),
                max_chars=int(payload.get("max_chars") or 6000),
                temperature=float(payload.get("temperature") or 0.0),
                top_p=float(payload.get("top_p") or 1.0),
                seed=int(payload.get("seed") or 42),
            )
        elif job_type == "e2e":
            from ..agents.crewai_impl import run_crewai_end_to_end_async
            await run_crewai_end_to_end_async(
                query=payload.get("query") or "",
                work_dir=Path(payload.get("work_dir") or "work/e2e"),
                html_dir=Path(payload.get("html_dir") or "pages/e2e"),
                db_path=Path(payload.get("db_path") or "artifacts/e2e.db"),
                out_jsonl=Path(payload.get("out_jsonl") or "artifacts/e2e_summaries.jsonl"),
                api_config_path=payload.get("api_config_path"),
                provider=payload.get("provider") or "ollama",
                base_url=payload.get("base_url") or "http://localhost:11434",
                api_key=payload.get("api_key"),
                model=payload.get("model") or "gpt-oss:20b",
                limit=int(payload.get("limit") or 5),
                max_results=int(payload.get("max_results") or 50),
                max_chars=int(payload.get("max_chars") or 12000),
                temperature=float(payload.get("temperature") or 0.0),
                top_p=float(payload.get("top_p") or 1.0),
                seed=int(payload.get("seed") or 42),
            )
        else:
            raise ValueError(f"Unknown job type: {job_type}")
        conn2 = qdb.connect()
        conn2.execute("UPDATE jobs SET status='done', finished_at=? WHERE id=?", (time.time(), job_id))
        conn2.commit()
        conn2.close()
    except Exception as e:
        conn3 = qdb.connect()
        conn3.execute(
            "UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
            (time.time(), str(e), job_id),
        )
        conn3.commit()
        conn3.close()
    return job_id


async def _agent_do(payload: Dict[str, Any]) -> None:
    query = payload.get("query") or ""
    provider = payload.get("provider") or "serpapi"
    api_config_path = payload.get("api_config_path")
    max_results = int(payload.get("max_results") or 5)
    exclude_domains = payload.get("exclude_domains") or []
    save_dir = Path(payload.get("save_html_dir") or "pages/agent")
    save_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    settings.search_provider = provider  # type: ignore
    if api_config_path:
        settings.api_config_path = api_config_path  # type: ignore
    api_cfg = load_api_config(settings.api_config_path)
    client = get_search_client(settings, api_cfg)
    # Fetch with pagination to reach max_results
    results = []
    offset = 0
    remaining = max_results
    per_page = 10
    while remaining > 0:
        n = min(per_page, remaining)
        batch = await client.search(query, count=n, offset=offset)
        if not batch:
            break
        results.extend(batch)
        got = len(batch)
        remaining -= got
        offset += got
        if got < n:
            break
    if exclude_domains:
        import tldextract

        ex = set(exclude_domains)
        fil = []
        for r in results:
            ext = tldextract.extract(r.url)
            domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
            if domain in ex:
                continue
            fil.append(r)
        results = fil
    urls = [r.url for r in results if r.url]
    async with Crawler(
        user_agent=settings.user_agent,
        request_timeout=settings.request_timeout,
        per_domain_delay=1.0,
        max_concurrency=4,
    ) as crawler:
        pages = await crawler.fetch_many(urls)
        for p in pages:
            if p.html:
                filename = (p.final_url or p.url).replace("://", "_")
                filename = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in filename)
                (save_dir / f"{filename}.html").write_text(p.html, encoding=p.encoding or "utf-8", errors="ignore")
