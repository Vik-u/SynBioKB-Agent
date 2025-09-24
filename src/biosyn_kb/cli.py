from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

import os
from pathlib import Path
from .search.clients import get_search_client
from .settings import get_settings
from .crawl import Crawler
from .extract import extract_main_text
from .extract.clean import extract_clean_text
from .extract.pdf import extract_pdf_text
from .entities import extract_entities
from .llm import list_ollama_models
from .llm.config import load_llm_config
from .summarize import summarize_pages_file
from .store import init_db, import_summaries_jsonl, query_metrics, import_pages_from_html_dir, retrieve_paragraphs
from .rag.vector_store import build_or_update_index, query_index
from .pdf import find_pdf_links_in_html, download_pdfs_from_urls
from .agents import run_local_pipeline
from .agents import run_crewai_or_fallback
from .api_config import load_api_config
from .search.open_access import enrich_urls_with_oa
from .queue import QueueDB, enqueue_job, run_worker_once, run_worker_once_async


async def cmd_search(args: argparse.Namespace) -> int:
    settings = get_settings()
    # Allow overriding provider from CLI
    if args.provider:
        # Inject override without rebuilding Settings model entirely
        settings.search_provider = args.provider  # type: ignore[attr-defined]
    if args.serpapi_key:
        settings.serpapi_api_key = args.serpapi_key  # type: ignore[attr-defined]
    if hasattr(args, "api_config") and args.api_config:
        settings.api_config_path = args.api_config  # type: ignore[attr-defined]

    api_cfg = load_api_config(settings.api_config_path)
    client = get_search_client(settings, api_cfg)
    # Fetch results with simple pagination to reach --max-results
    async def _fetch_all() -> list:
        out = []
        offset = args.offset or 0
        remaining = args.max_results
        per_page = 10  # safe default for providers
        while remaining > 0:
            n = min(per_page, remaining)
            batch = await client.search(args.query, count=n, offset=offset)
            if not batch:
                break
            out.extend(batch)
            got = len(batch)
            remaining -= got
            offset += got
            if got < n:
                break
        return out
    if hasattr(client, "__aenter__"):
        async with client:  # type: ignore[attr-defined]
            results = await _fetch_all()
    else:
        results = await _fetch_all()

    # Exclude domains if requested
    if args.exclude_domain:
        import tldextract

        exclude = set(args.exclude_domain)
        filtered = []
        for r in results:
            ext = tldextract.extract(r.url)
            domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
            if domain in exclude:
                continue
            filtered.append(r)
        results = filtered

    for r in results:
        print(json.dumps({
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "rank": r.rank,
            "source": r.source,
        }, ensure_ascii=False))
    return 0


def _iter_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    if args.url:
        urls.extend(args.url)
    if args.urls_file:
        if args.urls_file == "-":
            urls.extend([ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()])
        else:
            with open(args.urls_file, "r", encoding="utf-8") as fh:
                urls.extend([ln.strip() for ln in fh if ln.strip()])
    # Remove duplicates, preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


async def cmd_crawl(args: argparse.Namespace) -> int:
    urls = _iter_urls(args)
    if not urls:
        print("No URLs provided. Use --url or --urls-file.", file=sys.stderr)
        return 2

    settings = get_settings()
    out_dir: Path | None = None
    if args.save_html_dir:
        out_dir = Path(args.save_html_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    async with Crawler(
        user_agent=settings.user_agent,
        request_timeout=settings.request_timeout,
        per_domain_delay=args.per_domain_delay,
        max_concurrency=args.max_concurrency,
    ) as crawler:
        pages = await crawler.fetch_many(urls)
        for p in pages:
            if out_dir and p.html:
                filename = _safe_filename(p.final_url) + ".html"
                (out_dir / filename).write_text(p.html, encoding=p.encoding or "utf-8", errors="ignore")
            print(json.dumps({
                "url": p.url,
                "final_url": p.final_url,
                "status": p.status,
                "content_type": p.content_type,
                "encoding": p.encoding,
                "error": p.error,
            }, ensure_ascii=False))
    return 0


def _safe_filename(url: str) -> str:
    # Simple safe filename: strip scheme, replace non-alnum
    import re

    fn = url
    fn = fn.replace("://", "_")
    fn = re.sub(r"[^a-zA-Z0-9._-]+", "_", fn)
    return fn[:180]


async def cmd_run_all(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.provider:
        settings.search_provider = args.provider  # type: ignore[attr-defined]
    if args.serpapi_key:
        settings.serpapi_api_key = args.serpapi_key  # type: ignore[attr-defined]
    if hasattr(args, "api_config") and args.api_config:
        settings.api_config_path = args.api_config  # type: ignore[attr-defined]

    api_cfg = load_api_config(settings.api_config_path)
    client = get_search_client(settings, api_cfg)
    async def _fetch_all() -> list:
        out = []
        offset = args.offset or 0
        remaining = args.max_results
        per_page = 10
        while remaining > 0:
            n = min(per_page, remaining)
            batch = await client.search(args.query, count=n, offset=offset)
            if not batch:
                break
            out.extend(batch)
            got = len(batch)
            remaining -= got
            offset += got
            if got < n:
                break
        return out
    if hasattr(client, "__aenter__"):
        async with client:  # type: ignore[attr-defined]
            results = await _fetch_all()
    else:
        results = await _fetch_all()

    # Exclude domains if requested
    if args.exclude_domain:
        import tldextract

        exclude = set(args.exclude_domain)
        filtered = []
        for r in results:
            ext = tldextract.extract(r.url)
            domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
            if domain in exclude:
                continue
            filtered.append(r)
        results = filtered

    urls = [r.url for r in results if r.url]
    # Ethical OA enrichment via Unpaywall when DOI is present
    urls = await enrich_urls_with_oa(urls, api_cfg)

    out_dir: Path | None = None
    if args.save_html_dir:
        out_dir = Path(args.save_html_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    async with Crawler(
        user_agent=settings.user_agent,
        request_timeout=settings.request_timeout,
        per_domain_delay=args.per_domain_delay,
        max_concurrency=args.max_concurrency,
    ) as crawler:
        pages = await crawler.fetch_many(urls)
        for p in pages:
            if out_dir and p.html:
                filename = _safe_filename(p.final_url) + ".html"
                (out_dir / filename).write_text(p.html, encoding=p.encoding or "utf-8", errors="ignore")
            print(json.dumps({
                "url": p.url,
                "final_url": p.final_url,
                "status": p.status,
                "content_type": p.content_type,
                "encoding": p.encoding,
                "error": p.error,
            }, ensure_ascii=False))
    return 0


async def cmd_extract(args: argparse.Namespace) -> int:
    html_dir = Path(args.html_dir)
    if not html_dir.is_dir():
        print(f"Not a directory: {html_dir}", file=sys.stderr)
        return 2
    sink = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        for path in sorted(html_dir.glob("*.html")):
            html = path.read_text(encoding="utf-8", errors="ignore")
            if args.clean:
                content = extract_clean_text(html, url=None)
                title = content.title
                text = content.text
            else:
                content = extract_main_text(html)
                title = content.title
                text = content.text
            if args.max_chars and args.max_chars > 0:
                text = text[: args.max_chars]
            rec = {
                "file": str(path),
                "title": title,
                "text": text,
            }
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if args.include_pdfs:
            for p in sorted(html_dir.glob("*.pdf")):
                pdfc = extract_pdf_text(str(p))
                t = pdfc.text or ""
                if args.max_chars and args.max_chars > 0:
                    t = t[: args.max_chars]
                rec = {"file": str(p), "title": pdfc.title, "text": t}
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if sink is not sys.stdout:
            sink.close()
    return 0


async def cmd_entities(args: argparse.Namespace) -> int:
    import json

    # Default to LLM-based extraction unless explicitly disabled
    use_llm = not getattr(args, "no_llm", False)
    sink = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        with (open(args.inp, "r", encoding="utf-8") if args.inp != "-" else sys.stdin) as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or ""
                if use_llm:
                    from .entities import llm_extract_entities_text

                    data = await llm_extract_entities_text(
                        text,
                        model=args.model or "gpt-oss:20b",
                        ollama_url=args.ollama_url,
                        seed_chemical=args.chemical,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                    )
                    rec = {
                        "file": obj.get("file"),
                        "title": obj.get("title"),
                        "chemical": data.get("chemical") or args.chemical,
                        "enzymes_ec": data.get("enzymes_ec") or [],
                        "organisms": data.get("organisms") or [],
                        "yields_percent": data.get("yields_percent") or [],
                        "evidence": data.get("evidence"),
                        "method": "llm",
                    }
                else:
                    ext = extract_entities(text, seed_chemical=args.chemical)
                    rec = {
                        "file": obj.get("file"),
                        "title": obj.get("title"),
                        "chemical": ext.chemical,
                        "enzymes_ec": ext.enzymes_ec,
                        "organisms": ext.organisms,
                        "yields_percent": ext.yields_percent,
                        "evidence": ext.evidence,
                        "confidence": ext.confidence,
                        "method": "regex",
                    }
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if sink is not sys.stdout:
            sink.close()
    return 0


async def cmd_llm_models(args: argparse.Namespace) -> int:
    try:
        models = await list_ollama_models()
    except Exception as e:
        print(f"Error listing Ollama models: {e}", file=sys.stderr)
        return 1
    for m in models:
        print(m)
    return 0


async def cmd_summarize(args: argparse.Namespace) -> int:
    inp = Path(args.inp)
    out = Path(args.out)
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2
    # Load LLM config if provided or if llm.yaml exists in CWD
    cfg_path = getattr(args, "llm_config", None)
    if not cfg_path:
        default_cfg = Path("llm.yaml")
        if default_cfg.is_file():
            cfg_path = str(default_cfg)
    if cfg_path:
        cfg = load_llm_config(cfg_path)
        if cfg:
            if not args.model and cfg.model:
                args.model = cfg.model
            # provider/base_url/api_key support (online providers)
            if getattr(args, "provider", None) in (None, "ollama") and cfg.provider:
                args.provider = cfg.provider
            if getattr(args, "base_url", None) in (None, "http://localhost:11434") and cfg.base_url:
                args.base_url = cfg.base_url
            if getattr(args, "api_key", None) in (None, "") and cfg.api_key:
                args.api_key = cfg.api_key
            if args.temperature == 0.0 and cfg.temperature is not None:
                args.temperature = cfg.temperature
            if args.top_p == 1.0 and cfg.top_p is not None:
                args.top_p = cfg.top_p
            if args.seed == 42 and cfg.seed is not None:
                args.seed = cfg.seed

    # Auto-pick model if not provided: choose the model with the largest parameter size if parseable, else first
    model = args.model
    if not model:
        try:
            models = await list_ollama_models(args.ollama_url)
            # Try to pick the largest 20b/30b/70b etc.
            def sort_key(name: str) -> int:
                import re

                m = re.search(r"(\d+)(b|B)", name)
                return int(m.group(1)) if m else 0

            if models:
                models_sorted = sorted(models, key=sort_key, reverse=True)
                model = models_sorted[0]
        except Exception:
            pass
    if not model:
        print("No Ollama model specified and none detected. Pass --model.", file=sys.stderr)
        return 2

    await summarize_pages_file(
        inp,
        out,
        model=model,
        provider=getattr(args, "provider", "ollama"),
        base_url=getattr(args, "base_url", args.ollama_url),
        api_key=getattr(args, "api_key", None),
        limit=args.limit,
        concurrency=args.concurrency,
        max_chars=args.max_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        use_filter=args.filter,
        chunked=args.chunked,
        chunk_chars=args.chunk_chars,
    )
    return 0


async def cmd_db_init(args: argparse.Namespace) -> int:
    init_db(args.db)
    print(f"Initialized DB at {args.db}")
    return 0


async def cmd_db_import(args: argparse.Namespace) -> int:
    count = import_summaries_jsonl(args.db, args.inp)
    print(f"Imported {count} summaries into {args.db}")
    return 0


async def cmd_db_query(args: argparse.Namespace) -> int:
    rows = query_metrics(args.db, chemical=args.chemical, limit=args.limit)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    return 0


async def cmd_db_pages(args: argparse.Namespace) -> int:
    count = import_pages_from_html_dir(
        args.db,
        args.html_dir,
        use_cleaner=(not args.no_clean),
        include_pdfs=bool(args.include_pdfs),
        pdf_dir=args.pdf_dir,
    )
    print(f"Imported {count} pages into {args.db}")
    return 0


async def cmd_pdf_download(args: argparse.Namespace) -> int:
    urls: list[str] = []
    if args.urls_file:
        with open(args.urls_file, "r", encoding="utf-8") as fh:
            for ln in fh:
                u = ln.strip()
                if u:
                    urls.append(u)
    if args.html_dir:
        for p in Path(args.html_dir).glob("*.html"):
            html = p.read_text(encoding="utf-8", errors="ignore")
            pdfs = find_pdf_links_in_html(html)
            urls.extend(pdfs)
    # Dedup
    seen = set()
    uniq = [u for u in urls if not (u in seen or seen.add(u))]
    results = await download_pdfs_from_urls(uniq, args.out_dir)
    for src, path in results:
        print(json.dumps({"source": src, "pdf": path}))
    return 0


async def cmd_pipeline_run(args: argparse.Namespace) -> int:
    from .agents.orchestrator import PipelineConfig

    # Load LLM config if provided or if llm.yaml exists
    cfg_path = getattr(args, "llm_config", None)
    if not cfg_path:
        default_cfg = Path("llm.yaml")
        if default_cfg.is_file():
            cfg_path = str(default_cfg)
    if cfg_path:
        cfg = load_llm_config(cfg_path)
        if cfg:
            if args.model == "gpt-oss:20b" and cfg.model:
                args.model = cfg.model
            if args.ollama_url == "http://localhost:11434" and cfg.base_url:
                args.ollama_url = cfg.base_url
            if args.temperature == 0.0 and cfg.temperature is not None:
                args.temperature = cfg.temperature
            if args.top_p == 1.0 and cfg.top_p is not None:
                args.top_p = cfg.top_p
            if args.seed == 42 and cfg.seed is not None:
                args.seed = cfg.seed

    cfg = PipelineConfig(
        html_dir=Path(args.html_dir),
        work_dir=Path(args.work_dir),
        db_path=Path(args.db),
        summaries_jsonl=Path(args.out),
        model=args.model,
        ollama_url=args.ollama_url,
        limit=args.limit,
        max_chars=args.max_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        skip_llm=bool(args.skip_llm),
    )
    await run_local_pipeline(cfg)
    print(json.dumps({"status": "ok", "db": args.db, "summaries": args.out}))
    return 0


async def cmd_crewai_run(args: argparse.Namespace) -> int:
    await run_crewai_or_fallback(
        html_dir=Path(args.html_dir),
        work_dir=Path(args.work_dir),
        db_path=Path(args.db),
        out_jsonl=Path(args.out),
        model=args.model,
        ollama_url=args.ollama_url,
        limit=args.limit,
        max_chars=args.max_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(json.dumps({"status": "ok", "db": args.db, "summaries": args.out}))
    return 0


async def cmd_crewai_e2e(args: argparse.Namespace) -> int:
    # End-to-end CrewAI: Strategist -> Search+OA -> Crawl -> Extract -> Summarize -> Store -> Compose
    from .agents.crewai_impl import run_crewai_end_to_end_async
    await run_crewai_end_to_end_async(
        query=args.query,
        work_dir=Path(args.work_dir),
        html_dir=Path(args.html_dir),
        db_path=Path(args.db),
        out_jsonl=Path(args.out),
        api_config_path=getattr(args, "api_config", None),
        provider=getattr(args, "provider", "ollama"),
        base_url=getattr(args, "base_url", "http://localhost:11434"),
        api_key=getattr(args, "api_key", None),
        model=args.model,
        limit=args.limit,
        max_results=args.max_results,
        max_chars=args.max_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(json.dumps({"status": "ok", "db": args.db, "summaries": args.out, "report": str(Path(args.work_dir)/"report.md")}))
    return 0


async def cmd_qa(args: argparse.Namespace) -> int:
    from .llm.factory import get_llm_client
    # Prefer vector-store retrieval; build index if missing
    rag_dir = Path(getattr(args, "rag_dir", "artifacts/rag"))
    if not rag_dir.exists() or not any(rag_dir.iterdir()):
        build_or_update_index(args.db, rag_dir)
    ctxs = query_index(rag_dir, args.question, k=args.k)
    if not ctxs:
        ctxs = retrieve_paragraphs(args.db, args.question, k=args.k)
    if not ctxs:
        print(json.dumps({"answer": None, "contexts": []}))
        return 0
    prompt = (
        "You are a domain expert. Using ONLY the provided context paragraphs, answer the question."
        " Cite the source URLs inline as [n]. If unknown, say you don't know.\n\n"
        f"Question: {args.question}\n\n"
        "Contexts:\n"
    )
    for i, c in enumerate(ctxs, 1):
        prompt += f"[{i}] {c['para']}\nSource: {c['url']}\n\n"
    prompt += "Answer:"\

    async with get_llm_client(getattr(args, "provider", "ollama"), getattr(args, "base_url", args.ollama_url), getattr(args, "api_key", None)) as oc:
        out = await oc.generate(args.model, prompt, options={"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed})
    print(json.dumps({"answer": out.strip(), "contexts": ctxs}, ensure_ascii=False))
    return 0


async def cmd_agent_query(args: argparse.Namespace) -> int:
    # 1) Search using provider
    settings = get_settings()
    if args.provider:
        settings.search_provider = args.provider  # type: ignore[attr-defined]
    if hasattr(args, "api_config") and args.api_config:
        settings.api_config_path = args.api_config  # type: ignore[attr-defined]

    # Load API config
    api_cfg = load_api_config(settings.api_config_path)
    client = get_search_client(settings, api_cfg)
    async def _fetch_all() -> list:
        out = []
        offset = 0
        remaining = args.max_results
        per_page = 10
        while remaining > 0:
            n = min(per_page, remaining)
            batch = await client.search(args.query, count=n, offset=offset)
            if not batch:
                break
            out.extend(batch)
            got = len(batch)
            remaining -= got
            offset += got
            if got < n:
                break
        return out
    if hasattr(client, "__aenter__"):
        async with client:  # type: ignore[attr-defined]
            results = await _fetch_all()
    else:
        results = await _fetch_all()
    # Domain exclude
    if args.exclude_domain:
        import tldextract

        exclude = set(args.exclude_domain)
        filtered = []
        for r in results:
            ext = tldextract.extract(r.url)
            domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
            if domain in exclude:
                continue
            filtered.append(r)
        results = filtered

    urls = [r.url for r in results if r.url]
    urls = await enrich_urls_with_oa(urls, api_cfg)
    # 2) Crawl
    out_dir = Path(args.save_html_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
                (out_dir / f"{filename}.html").write_text(p.html, encoding=p.encoding or "utf-8", errors="ignore")
    # 3) Optional: zero-shot primer using LLM
    try:
        from .llm.ollama_client import OllamaClient

        primer_prompt = (
            "You are a biosynthesis expert. Provide a concise primer for the following topic,"
            " covering pathways, key enzymes (with EC numbers when known), likely organisms,"
            " feedstocks, and typical performance ranges (yield/titer/productivity)."
            " Use cautious language and include 5-8 bullet points. If uncertain, say so.\n\n"
            f"Topic: {args.query}\n\nPrimer:"
        )
        async with OllamaClient(args.ollama_url) as oc:
            primer = await oc.generate(args.model, primer_prompt, options={"temperature": max(0.2, args.temperature), "top_p": args.top_p, "seed": args.seed})
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.work_dir) / "primer.txt").write_text(primer, encoding="utf-8")
    except Exception:
        pass

    # 4) Crew pipeline on saved HTML
    # Load LLM defaults if provided/available
    cfg_path = getattr(args, "llm_config", None)
    if not cfg_path:
        default_cfg = Path("llm.yaml")
        if default_cfg.is_file():
            cfg_path = str(default_cfg)
    if cfg_path:
        llm_cfg = load_llm_config(cfg_path)
        if llm_cfg:
            if args.model == "gpt-oss:20b" and llm_cfg.model:
                args.model = llm_cfg.model
            if args.ollama_url == "http://localhost:11434" and llm_cfg.base_url:
                args.ollama_url = llm_cfg.base_url
            if args.temperature == 0.0 and llm_cfg.temperature is not None:
                args.temperature = llm_cfg.temperature
            if args.top_p == 1.0 and llm_cfg.top_p is not None:
                args.top_p = llm_cfg.top_p
            if args.seed == 42 and llm_cfg.seed is not None:
                args.seed = llm_cfg.seed

    await run_crewai_or_fallback(
        html_dir=out_dir,
        work_dir=Path(args.work_dir),
        db_path=Path(args.db),
        out_jsonl=Path(args.out),
        model=args.model,
        ollama_url=args.ollama_url,
        limit=args.limit,
        max_chars=args.max_chars,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(json.dumps({"status": "ok", "html_dir": str(out_dir), "db": args.db, "summaries": args.out}, ensure_ascii=False))
    return 0


async def cmd_qadd_crawl(args: argparse.Namespace) -> int:
    urls = []
    with open(args.urls_file, "r", encoding="utf-8") as fh:
        for ln in fh:
            u = ln.strip()
            if u:
                urls.append(u)
    q = QueueDB(Path(args.queue_db))
    job_id = enqueue_job(q, "crawl", {"urls": urls, "save_html_dir": args.save_html_dir})
    print(json.dumps({"enqueued": job_id}))
    return 0


async def cmd_qadd_summarize(args: argparse.Namespace) -> int:
    q = QueueDB(Path(args.queue_db))
    payload = {
        "input_jsonl": args.inp,
        "output_jsonl": args.out,
        "model": args.model,
        "ollama_url": args.ollama_url,
        "limit": args.limit,
        "max_chars": args.max_chars,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "use_filter": True,
        "chunked": True,
        "chunk_chars": 2800,
    }
    job_id = enqueue_job(q, "summarize", payload)
    print(json.dumps({"enqueued": job_id}))
    return 0


async def cmd_qworker_once(args: argparse.Namespace) -> int:
    q = QueueDB(Path(args.queue_db))
    # Prefer async worker to avoid nested event loop issues
    job_id = await run_worker_once_async(q)
    print(json.dumps({"processed": job_id}))
    return 0


async def cmd_qadd_agent(args: argparse.Namespace) -> int:
    ex = getattr(args, "exclude_domain", None) or []
    q = QueueDB(Path(args.queue_db))
    payload = {
        "query": args.query,
        "provider": args.provider,
        "api_config_path": args.api_config,
        "max_results": args.max_results,
        "exclude_domains": ex,
        "save_html_dir": args.save_html_dir,
    }
    job_id = enqueue_job(q, "agent", payload)
    print(json.dumps({"enqueued": job_id}))
    return 0


async def cmd_qworker_watch(args: argparse.Namespace) -> int:
    import asyncio as _asyncio
    q = QueueDB(Path(args.queue_db))
    try:
        while True:
            job_id = await run_worker_once_async(q)
            if job_id is None:
                await _asyncio.sleep(args.sleep)
            else:
                # yield to event loop briefly
                await _asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print(json.dumps({"status": "stopped"}))
        return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biosyn-kb", description="Biosynthesis KB Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Run a web search and print results as JSON lines")
    p_search.add_argument("--query", required=True, help="Search query string")
    p_search.add_argument("--max-results", type=int, default=10, help="Number of results to fetch")
    p_search.add_argument("--offset", type=int, default=0, help="Results offset for pagination")
    p_search.add_argument("--provider", choices=["serpapi", "brave"], help="Search provider override")
    p_search.add_argument("--serpapi-key", help="SerpAPI API key (optional override)")
    p_search.add_argument("--api-config", help="Path to apis.yaml config file")
    p_search.add_argument("--exclude-domain", action="append", help="Exclude domain (e.g., wikipedia.org). Repeatable.")
    p_search.set_defaults(func=cmd_search)

    p_crawl = sub.add_parser("crawl", help="Fetch pages from URLs (robots-aware) and print summaries as JSON lines")
    p_crawl.add_argument("--url", action="append", help="URL to fetch (can be passed multiple times)")
    p_crawl.add_argument("--urls-file", help="Path to a file with newline-separated URLs; use '-' for stdin")
    p_crawl.add_argument("--save-html-dir", help="Directory to save fetched HTML files (optional)")
    p_crawl.add_argument("--per-domain-delay", type=float, default=1.0, help="Delay between requests per domain (seconds)")
    p_crawl.add_argument("--max-concurrency", type=int, default=8, help="Max concurrent requests")
    p_crawl.add_argument("--headless", action="store_true", help="Use headless browser (Playwright) for JS pages")
    p_crawl.set_defaults(func=cmd_crawl)

    p_run = sub.add_parser("run-all", help="Run search then crawl results")
    p_run.add_argument("--query", required=True, help="Search query string")
    p_run.add_argument("--max-results", type=int, default=10, help="Number of results to fetch")
    p_run.add_argument("--offset", type=int, default=0, help="Results offset for pagination")
    p_run.add_argument("--provider", choices=["serpapi", "brave"], help="Search provider override")
    p_run.add_argument("--serpapi-key", help="SerpAPI API key (optional override)")
    p_run.add_argument("--api-config", help="Path to apis.yaml config file")
    p_run.add_argument("--exclude-domain", action="append", help="Exclude domain (e.g., wikipedia.org). Repeatable.")
    p_run.add_argument("--save-html-dir", help="Directory to save fetched HTML files (optional)")
    p_run.add_argument("--per-domain-delay", type=float, default=1.0, help="Delay between requests per domain (seconds)")
    p_run.add_argument("--max-concurrency", type=int, default=8, help="Max concurrent requests")
    p_run.set_defaults(func=cmd_run_all)

    p_extract = sub.add_parser("extract", help="Extract main text from saved HTML files in a directory")
    p_extract.add_argument("--html-dir", required=True, help="Directory containing .html files (from crawl)")
    p_extract.add_argument("--out", help="Write JSONL output to this file (default: stdout)")
    p_extract.add_argument("--max-chars", type=int, default=0, help="Truncate extracted text to this many chars (0 = no limit)")
    p_extract.add_argument("--clean", action="store_true", help="Use advanced cleaner (trafilatura) for boilerplate removal")
    p_extract.add_argument("--include-pdfs", action="store_true", help="Also include .pdf files from the same directory")
    p_extract.set_defaults(func=cmd_extract)

    p_entities = sub.add_parser("entities", help="Extract biosynthesis entities from extracted text JSONL (LLM by default)")
    p_entities.add_argument("--in", dest="inp", required=True, help="Input JSONL from 'extract' step or similar")
    p_entities.add_argument("--chemical", help="Seed chemical name (optional)")
    p_entities.add_argument("--out", help="Write JSONL output to this file (default: stdout)")
    # LLM options (default enabled)
    p_entities.add_argument("--no-llm", action="store_true", help="Disable LLM and use regex heuristics instead")
    p_entities.add_argument("--model", default="gpt-oss:20b", help="Ollama model for entity extraction")
    p_entities.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    p_entities.add_argument("--provider", default="ollama", help="LLM provider: ollama|openai")
    p_entities.add_argument("--base-url", default="http://localhost:11434", help="LLM base URL")
    p_entities.add_argument("--api-key", help="API key for online providers")
    p_entities.add_argument("--temperature", type=float, default=0.0)
    p_entities.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    p_entities.add_argument("--seed", type=int, default=42)
    p_entities.set_defaults(func=cmd_entities)

    p_llm = sub.add_parser("llm-models", help="List installed Ollama models")
    p_llm.set_defaults(func=cmd_llm_models)

    p_sum = sub.add_parser("summarize", help="Summarize extracted pages JSONL into structured records using LLMs (Ollama default)")
    p_sum.add_argument("--in", dest="inp", required=True, help="Input JSONL from 'extract' step")
    p_sum.add_argument("--out", required=True, help="Output JSONL for structured summaries")
    p_sum.add_argument("--model", help="Ollama model name (defaults to largest installed)")
    p_sum.add_argument("--ollama-url", default="http://localhost:11434", help="[DEPRECATED name] Ollama base URL; use --base-url")
    p_sum.add_argument("--provider", default="ollama", help="LLM provider: ollama|openai")
    p_sum.add_argument("--base-url", default="http://localhost:11434", help="LLM base URL (Ollama or OpenAI-compatible)")
    p_sum.add_argument("--api-key", help="API key for online providers (e.g., OpenAI)")
    p_sum.add_argument("--limit", type=int, help="Limit number of pages to summarize")
    p_sum.add_argument("--concurrency", type=int, default=2, help="Concurrent LLM requests")
    p_sum.add_argument("--max-chars", type=int, default=12000, help="Max characters from each page to feed the model")
    p_sum.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (default 0.0 for determinism)")
    p_sum.add_argument("--top-p", dest="top_p", type=float, default=1.0, help="LLM top-p (default 1.0)")
    p_sum.add_argument("--seed", type=int, default=42, help="LLM seed for reproducibility (default 42)")
    p_sum.add_argument("--filter", action="store_true", help="Apply relevance filter before summarization")
    p_sum.add_argument("--chunked", action="store_true", help="Enable chunked summarization for long pages")
    p_sum.add_argument("--chunk-chars", type=int, default=3500, help="Chunk size in characters when --chunked is used")
    p_sum.set_defaults(func=cmd_summarize)

    # DB commands
    p_db_init = sub.add_parser("db-init", help="Initialize SQLite database")
    p_db_init.add_argument("--db", default="biosyn.db", help="SQLite DB path (default: biosyn.db)")
    p_db_init.set_defaults(func=cmd_db_init)

    p_db_import = sub.add_parser("db-import-summaries", help="Import summaries JSONL into DB")
    p_db_import.add_argument("--db", default="biosyn.db", help="SQLite DB path")
    p_db_import.add_argument("--in", dest="inp", required=True, help="Summaries JSONL file")
    p_db_import.set_defaults(func=cmd_db_import)

    p_db_query = sub.add_parser("db-query-metrics", help="List metrics for a chemical from DB")
    p_db_query.add_argument("--db", default="biosyn.db", help="SQLite DB path")
    p_db_query.add_argument("--chemical", help="Filter by chemical name (exact)")
    p_db_query.add_argument("--limit", type=int, default=20, help="Max rows")
    p_db_query.set_defaults(func=cmd_db_query)

    p_db_pages = sub.add_parser("db-import-pages", help="Import raw/cleaned HTML pages into DB")
    p_db_pages.add_argument("--db", default="biosyn.db", help="SQLite DB path")
    p_db_pages.add_argument("--html-dir", required=True, help="Directory with .html files")
    p_db_pages.add_argument("--no-clean", action="store_true", help="Store without cleaner (raw only)")
    p_db_pages.add_argument("--include-pdfs", action="store_true", help="Also parse .pdf files in the same directory or --pdf-dir")
    p_db_pages.add_argument("--pdf-dir", help="Directory with PDFs (defaults to html-dir)")
    p_db_pages.set_defaults(func=cmd_db_pages)

    p_pdf = sub.add_parser("pdf-download", help="Find and download PDFs from HTML files or URL list")
    p_pdf.add_argument("--html-dir", help="Directory containing .html files to scan for PDFs")
    p_pdf.add_argument("--urls-file", help="Text file with PDF URLs or page URLs (one per line)")
    p_pdf.add_argument("--out-dir", default="pdfs", help="Directory to save PDFs")
    p_pdf.set_defaults(func=cmd_pdf_download)

    # Pipeline (multi-agent orchestrator) demo command
    p_pipe = sub.add_parser("pipeline-run", help="Run local multi-stage pipeline on saved HTML (extract -> summarize -> store)")
    p_pipe.add_argument("--html-dir", required=True, help="Directory with .html files")
    p_pipe.add_argument("--work-dir", default="work", help="Working directory for intermediate files")
    p_pipe.add_argument("--db", default="biosyn_pipeline.db", help="SQLite DB path")
    p_pipe.add_argument("--out", default="summaries.jsonl", help="Output summaries JSONL")
    p_pipe.add_argument("--model", default="gpt-oss:20b", help="Ollama model to use")
    p_pipe.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    p_pipe.add_argument("--limit", type=int, help="Limit number of pages to summarize")
    p_pipe.add_argument("--max-chars", type=int, default=12000, help="Max characters per page to LLM")
    p_pipe.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    p_pipe.add_argument("--top-p", dest="top_p", type=float, default=1.0, help="LLM top-p")
    p_pipe.add_argument("--seed", type=int, default=42, help="LLM seed")
    p_pipe.add_argument("--llm-config", help="YAML file with LLM defaults (provider, model, base_url, temperature, top_p, seed)")
    p_pipe.add_argument("--skip-llm", action="store_true", help="Skip LLM summarization (for dry runs/tests)")
    p_pipe.set_defaults(func=cmd_pipeline_run)

    # CrewAI (or fallback) demo command
    p_crew = sub.add_parser("crewai-run", help="Run crew-based (or fallback) pipeline on saved HTML")
    p_crew.add_argument("--html-dir", required=True, help="Directory with .html files")
    p_crew.add_argument("--work-dir", default="work/crew", help="Working directory")
    p_crew.add_argument("--db", default="biosyn_crew.db", help="SQLite DB path")
    p_crew.add_argument("--out", default="summaries_crew.jsonl", help="Output summaries JSONL")
    p_crew.add_argument("--model", default="gpt-oss:20b", help="Ollama model to use")
    p_crew.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    p_crew.add_argument("--limit", type=int, help="Limit number of pages to summarize")
    p_crew.add_argument("--max-chars", type=int, default=12000, help="Max characters per page to LLM")
    p_crew.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    p_crew.add_argument("--top-p", dest="top_p", type=float, default=1.0, help="LLM top-p")
    p_crew.add_argument("--seed", type=int, default=42, help="LLM seed")
    p_crew.set_defaults(func=cmd_crewai_run)

    # CrewAI End-to-End from query with Strategist and Composer
    p_ce2e = sub.add_parser("crewai-e2e", help="End-to-end CrewAI: strategy→search→crawl→summarize→store→compose")
    p_ce2e.add_argument("--query", required=True, help="User natural-language query")
    p_ce2e.add_argument("--html-dir", default="pages/e2e", help="Directory to save HTML")
    p_ce2e.add_argument("--work-dir", default="work/e2e", help="Working directory")
    p_ce2e.add_argument("--db", default="artifacts/e2e.db", help="SQLite DB path")
    p_ce2e.add_argument("--out", default="artifacts/e2e_summaries.jsonl", help="Output summaries JSONL")
    p_ce2e.add_argument("--provider", default="ollama", help="LLM provider: ollama|openai")
    p_ce2e.add_argument("--base-url", default="http://localhost:11434", help="LLM base URL")
    p_ce2e.add_argument("--api-key", help="API key for online providers")
    p_ce2e.add_argument("--api-config", help="Path to apis.yaml with keys (SerpAPI/Brave/Unpaywall)")
    p_ce2e.add_argument("--model", default="gpt-oss:20b", help="LLM model")
    p_ce2e.add_argument("--limit", type=int, default=5, help="Limit pages to summarize")
    p_ce2e.add_argument("--max-results", type=int, default=50, help="Max search results to collect per subquery")
    p_ce2e.add_argument("--max-chars", type=int, default=12000, help="Max chars per page to LLM")
    p_ce2e.add_argument("--temperature", type=float, default=0.0)
    p_ce2e.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    p_ce2e.add_argument("--seed", type=int, default=42)
    p_ce2e.set_defaults(func=cmd_crewai_e2e)

    # Q&A command (naive RAG over pages, then Ollama answer)
    p_qa = sub.add_parser("qa", help="Answer a question using retrieved paragraphs (RAG) and an LLM (Ollama default)")
    p_qa.add_argument("--db", default="biosyn.db", help="SQLite DB path")
    p_qa.add_argument("--question", required=True, help="Question to answer")
    p_qa.add_argument("--k", type=int, default=5, help="Top-k paragraphs to retrieve")
    p_qa.add_argument("--model", default="gpt-oss:20b", help="Model")
    p_qa.add_argument("--ollama-url", default="http://localhost:11434", help="[DEPRECATED name] Ollama base URL; use --base-url")
    p_qa.add_argument("--provider", default="ollama", help="LLM provider: ollama|openai")
    p_qa.add_argument("--base-url", default="http://localhost:11434", help="LLM base URL")
    p_qa.add_argument("--api-key", help="API key for online providers")
    p_qa.add_argument("--temperature", type=float, default=0.0)
    p_qa.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    p_qa.add_argument("--seed", type=int, default=42)
    p_qa.set_defaults(func=cmd_qa)

    # Compose a consolidated report from multiple summaries
    from .report.compose import compose_report_from_jsonl  # local import for CLI registration only

    async def cmd_compose(args: argparse.Namespace) -> int:
        inp = Path(args.inp)
        out = Path(args.out)
        if not inp.is_file():
            print(f"Input not found: {inp}", file=sys.stderr)
            return 2
        await compose_report_from_jsonl(
            inp,
            out,
            provider=getattr(args, "provider", "ollama"),
            base_url=getattr(args, "base_url", "http://localhost:11434"),
            api_key=getattr(args, "api_key", None),
            model=args.model,
        )
        print(json.dumps({"status": "ok", "report": str(out)}))
        return 0

    p_rep = sub.add_parser("compose-report", help="Compose a single consolidated Markdown report from summaries JSONL")
    p_rep.add_argument("--in", dest="inp", required=True, help="Summaries JSONL input")
    p_rep.add_argument("--out", required=True, help="Output Markdown path")
    p_rep.add_argument("--provider", default="ollama", help="LLM provider: ollama|openai")
    p_rep.add_argument("--base-url", default="http://localhost:11434", help="LLM base URL")
    p_rep.add_argument("--api-key", help="API key for online providers")
    p_rep.add_argument("--model", default="gpt-oss:20b", help="Model name")
    p_rep.set_defaults(func=cmd_compose)

    # Agentic end-to-end from a chat-like query: search -> crawl -> extract -> summarize -> store
    p_agent = sub.add_parser("agent-query", help="Run full pipeline from a conversational query (search + crawl + crew pipeline)")
    p_agent.add_argument("--query", required=True, help="Conversational query, e.g., 'biosynthesis of isobutanol'")
    p_agent.add_argument("--provider", choices=["serpapi", "brave"], default="serpapi", help="Search provider")
    p_agent.add_argument("--max-results", type=int, default=5, help="Number of results to fetch")
    p_agent.add_argument("--exclude-domain", action="append", help="Domain to exclude (repeatable)")
    p_agent.add_argument("--api-config", help="Path to apis.yaml config file")
    p_agent.add_argument("--save-html-dir", default="pages/agent", help="Where to save fetched HTML")
    p_agent.add_argument("--work-dir", default="work/agent", help="Working directory")
    p_agent.add_argument("--db", default="artifacts/agent.db", help="SQLite DB path")
    p_agent.add_argument("--out", default="artifacts/agent_summaries.jsonl", help="Output summaries JSONL")
    p_agent.add_argument("--model", default="gpt-oss:20b", help="Ollama model")
    p_agent.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    p_agent.add_argument("--limit", type=int, default=None, help="Limit number of pages summarized")
    p_agent.add_argument("--max-chars", type=int, default=6000, help="Max characters per page to LLM")
    p_agent.add_argument("--temperature", type=float, default=0.0)
    p_agent.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    p_agent.add_argument("--seed", type=int, default=42)
    p_agent.add_argument("--llm-config", help="YAML file with LLM defaults")
    p_agent.set_defaults(func=cmd_agent_query)

    # Queue commands
    p_qadd_crawl = sub.add_parser("queue-add-crawl", help="Enqueue a crawl job")
    p_qadd_crawl.add_argument("--queue-db", default="artifacts/queue.db", help="Queue DB path")
    p_qadd_crawl.add_argument("--urls-file", required=True, help="File with URLs to crawl (one per line)")
    p_qadd_crawl.add_argument("--save-html-dir", default="pages/queue", help="Where to save HTML")
    p_qadd_crawl.set_defaults(func=cmd_qadd_crawl)

    p_qadd_sum = sub.add_parser("queue-add-summarize", help="Enqueue a summarize job")
    p_qadd_sum.add_argument("--queue-db", default="artifacts/queue.db")
    p_qadd_sum.add_argument("--in", dest="inp", required=True)
    p_qadd_sum.add_argument("--out", required=True)
    p_qadd_sum.add_argument("--model", default="gpt-oss:20b")
    p_qadd_sum.add_argument("--ollama-url", default="http://localhost:11434")
    p_qadd_sum.add_argument("--limit", type=int)
    p_qadd_sum.add_argument("--max-chars", type=int, default=6000)
    p_qadd_sum.add_argument("--temperature", type=float, default=0.0)
    p_qadd_sum.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    p_qadd_sum.add_argument("--seed", type=int, default=42)
    p_qadd_sum.set_defaults(func=cmd_qadd_summarize)

    p_qworker = sub.add_parser("queue-worker-once", help="Run one queue job if available")
    p_qworker.add_argument("--queue-db", default="artifacts/queue.db")
    p_qworker.set_defaults(func=cmd_qworker_once)

    p_qadd_agent = sub.add_parser("queue-add-agent", help="Enqueue an agent job (search + crawl)")
    p_qadd_agent.add_argument("--queue-db", default="artifacts/queue.db")
    p_qadd_agent.add_argument("--query", required=True)
    p_qadd_agent.add_argument("--provider", choices=["serpapi", "brave"], default="serpapi")
    p_qadd_agent.add_argument("--api-config", help="apis.yaml path")
    p_qadd_agent.add_argument("--max-results", type=int, default=5)
    p_qadd_agent.add_argument("--exclude-domain", action="append")
    p_qadd_agent.add_argument("--save-html-dir", default="pages/agent")
    p_qadd_agent.set_defaults(func=cmd_qadd_agent)

    p_qwatch = sub.add_parser("queue-worker-watch", help="Run queue worker in a loop")
    p_qwatch.add_argument("--queue-db", default="artifacts/queue.db")
    p_qwatch.add_argument("--sleep", type=float, default=2.0, help="Seconds between polls")
    p_qwatch.set_defaults(func=cmd_qworker_watch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
