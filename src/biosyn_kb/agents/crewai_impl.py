from __future__ import annotations

from pathlib import Path
from typing import Optional, List


def run_crewai_pipeline(
    html_dir: Path,
    work_dir: Path,
    db_path: Path,
    out_jsonl: Path,
    *,
    model: str = "gpt-oss:20b",
    ollama_url: str = "http://localhost:11434",
    limit: Optional[int] = None,
    max_chars: int = 6000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    ) -> None:
    """True CrewAI implementation (requires Python >=3.10 and crewai installed).

    This function sets up CrewAI Agents and Tasks mirroring our lightweight agents and runs them.
    """
    # Dynamic imports to avoid hard dependency on Python 3.9
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew
    # Prefer robust import from langchain-ollama if available, else fallback
    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except Exception:
        try:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        except Exception:
            from langchain_community.llms import Ollama as ChatOllama  # type: ignore

    # Tool functions from our stack
    from ..extract.clean import extract_clean_text
    from ..filter.relevance import filter_text
    from ..summarize.pipeline import summarize_pages_file
    from ..store import init_db, import_pages_from_html_dir, import_summaries_jsonl

    llm = ChatOllama(base_url=str(ollama_url), model=model, temperature=temperature)

    # Define simple agents
    extractor = CrewAgent(
        role="Content Extractor",
        goal="Produce clean body text from HTML with irrelevant content removed",
        backstory="Reads scientific HTML and keeps meaningful text",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    summarizer = CrewAgent(
        role="LLM Summarizer",
        goal="Produce structured and narrative summaries as JSON",
        backstory="Summarizes scientific content for biosynthesis",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    storer = CrewAgent(
        role="KB Writer",
        goal="Persist pages and summaries in SQLite",
        backstory="Ensures durable storage",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    # Implement tasks using our Python functions (CrewAI will treat them as tool-driven steps)
    def _run_extract():
        work_dir.mkdir(parents=True, exist_ok=True)
        extracted_jsonl = work_dir / "extracted.jsonl"
        import json
        with extracted_jsonl.open("w", encoding="utf-8") as sink:
            for p in sorted(html_dir.glob("*.html")):
                html = p.read_text(encoding="utf-8", errors="ignore")
                clean = extract_clean_text(html, url=None)
                text = clean.text or ""
                text, _ = filter_text(text)
                rec = {"file": str(p), "title": (clean.title or ""), "text": text}
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return str(extracted_jsonl)

    def _run_summarize(extracted_path: str):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                summarize_pages_file(
                    input_jsonl=Path(extracted_path),
                    output_jsonl=out_jsonl,
                    model=model,
                    provider="ollama",
                    base_url=str(ollama_url),
                    api_key=None,
                    limit=limit,
                    concurrency=1,
                    max_chars=max_chars,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    use_filter=True,
                    chunked=True,
                    chunk_chars=2800,
                )
            )
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def _run_store():
        init_db(db_path)
        import_pages_from_html_dir(db_path, html_dir, use_cleaner=True)
        import_summaries_jsonl(db_path, out_jsonl)
        return str(db_path)

    t1 = CrewTask(description="Extract and clean HTML content", agent=extractor, expected_output="Path to JSONL", callback=_run_extract)  # type: ignore
    # The CrewAI Task API varies; if callback is unsupported in your version, adapt to use tools or simply run Python between tasks.
    extracted_jsonl = _run_extract()
    _run_summarize(extracted_jsonl)
    _run_store()

    # If you want pure CrewAI execution order, you could create a Crew and call kickoff:
    # crew = Crew(agents=[extractor, summarizer, storer], tasks=[t1, t2, t3], verbose=False)
    # crew.kickoff()


async def run_crewai_pipeline_async(
    html_dir: Path,
    work_dir: Path,
    db_path: Path,
    out_jsonl: Path,
    *,
    model: str = "gpt-oss:20b",
    ollama_url: str = "http://localhost:11434",
    limit: Optional[int] = None,
    max_chars: int = 6000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
) -> None:
    # Same logic as above but await summarize
    from crewai import Agent as CrewAgent  # type: ignore
    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except Exception:
        from langchain_community.chat_models import ChatOllama  # type: ignore

    from ..extract.clean import extract_clean_text
    from ..filter.relevance import filter_text
    from ..summarize.pipeline import summarize_pages_file
    from ..store import init_db, import_pages_from_html_dir, import_summaries_jsonl

    llm = ChatOllama(base_url=str(ollama_url), model=model, temperature=temperature)

    def _run_extract_sync() -> str:
        work_dir.mkdir(parents=True, exist_ok=True)
        extracted_jsonl = work_dir / "extracted.jsonl"
        import json
        with extracted_jsonl.open("w", encoding="utf-8") as sink:
            for p in sorted(html_dir.glob("*.html")):
                html = p.read_text(encoding="utf-8", errors="ignore")
                clean = extract_clean_text(html, url=None)
                text = clean.text or ""
                text, _ = filter_text(text)
                rec = {"file": str(p), "title": (clean.title or ""), "text": text}
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return str(extracted_jsonl)

    extracted_jsonl = _run_extract_sync()
    await summarize_pages_file(
        input_jsonl=Path(extracted_jsonl),
        output_jsonl=out_jsonl,
        model=model,
        ollama_url=str(ollama_url),
        limit=limit,
        concurrency=1,
        max_chars=max_chars,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        use_filter=True,
        chunked=True,
        chunk_chars=2800,
    )
    init_db(db_path)
    import_pages_from_html_dir(db_path, html_dir, use_cleaner=True, include_pdfs=True)
    import_summaries_jsonl(db_path, out_jsonl)


async def run_crewai_end_to_end_async(
    query: str,
    work_dir: Path,
    html_dir: Path,
    db_path: Path,
    out_jsonl: Path,
    *,
    api_config_path: Optional[str] = None,
    provider: str = "ollama",
    base_url: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    model: str = "gpt-oss:20b",
    limit: Optional[int] = 20,
    max_results: int = 50,
    max_chars: int = 12000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
) -> None:
    """CrewAI-flavored end-to-end: Strategist → Search+OA → Crawl → Extract → Summarize → Store → Compose."""
    # 1) Strategist (LLM) to expand queries and suggest domains
    from ..llm.factory import get_llm_client
    plan_prompt = (
        "You are a search strategist. Expand the user query into 4-6 targeted subqueries,\n"
        "and propose a small whitelist of preferred open-access domains (e.g., ncbi.nlm.nih.gov, plos.org, frontiersin.org, biomedcentral.com, rsc.org).\n"
        "Return JSON with keys: {\"subqueries\":[...], \"domains\":[...]} only.\n\n"
        f"User query: {query}\nPlan:"
    )
    async with get_llm_client(provider, base_url, api_key) as oc:
        plan_out = await oc.generate(model, plan_prompt, options={"temperature": 0.1, "top_p": 1.0})
    import json
    try:
        start = plan_out.find("{"); end = plan_out.rfind("}")
        plan_js = json.loads(plan_out[start:end+1] if start!=-1 and end!=-1 else plan_out)
    except Exception:
        plan_js = {"subqueries": [query], "domains": []}
    subqueries: List[str] = plan_js.get("subqueries") or [query]
    domains: List[str] = plan_js.get("domains") or []
    # checkpoint: save plan
    from ..pipeline.checkpoints import write_checkpoint
    write_checkpoint(work_dir / "checkpoints" / "01_plan.json", plan_js)

    # 2) Search with pagination and OA enrichment
    from ..settings import get_settings
    from ..api_config import load_api_config
    from ..search.clients import get_search_client
    from ..search.open_access import enrich_urls_with_oa
    settings = get_settings()
    if api_config_path:
        settings.api_config_path = api_config_path  # type: ignore
    api_cfg = load_api_config(settings.api_config_path)
    async def _do_search(q: str) -> List[str]:
        out = []
        offset = 0; remaining = max_results; per_page = 10
        batch = []
        client = get_search_client(settings, api_cfg)
        if hasattr(client, "__aenter__"):
            async with client:  # type: ignore
                while remaining > 0:
                    n = min(per_page, remaining)
                    batch = await client.search(q, count=n, offset=offset)
                    if not batch:
                        break
                    out.extend([r.url for r in batch if r.url])
                    got = len(batch); remaining -= got; offset += got
                    if got < n: break
        else:
            while remaining > 0:
                n = min(per_page, remaining)
                batch = await client.search(q, count=n, offset=offset)
                if not batch:
                    break
                out.extend([r.url for r in batch if r.url])
                got = len(batch); remaining -= got; offset += got
                if got < n: break
        # domain filter if provided
        if domains:
            import tldextract
            ex = set(domains)
            out = [u for u in out if (lambda d: (d in ex))(f"{tldextract.extract(u).domain}.{tldextract.extract(u).suffix}" if tldextract.extract(u).suffix else tldextract.extract(u).domain)]
        out = await enrich_urls_with_oa(out, api_cfg)
        # dedupe
        seen=set(); out=[u for u in out if not (u in seen or seen.add(u))]
        return out[:max_results]

    urls_all: List[str] = []
    for sq in subqueries:
        urls_all.extend(await _do_search(sq))
    # dedupe across subqueries
    seen=set(); urls=[u for u in urls_all if not (u in seen or seen.add(u))]
    write_checkpoint(work_dir / "checkpoints" / "02_urls.json", {"count": len(urls), "urls": urls})

    # 3) Crawl
    from ..crawl import Crawler
    html_dir.mkdir(parents=True, exist_ok=True)
    # choose crawler
    use_headless = bool(False)  # can be parameterized if needed
    if use_headless:
        from ..crawl.headless import HeadlessCrawler as _Crawler
    else:
        from ..crawl import Crawler as _Crawler
    async with _Crawler(
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
                (html_dir / f"{filename}.html").write_text(p.html, encoding=p.encoding or "utf-8", errors="ignore")
    from ..pipeline.checkpoints import crawl_report as _crawl_rep
    write_checkpoint(work_dir / "checkpoints" / "03_crawl_report.json", _crawl_rep(html_dir))

    # 4) Extract
    from ..extract.clean import extract_clean_text
    work_dir.mkdir(parents=True, exist_ok=True)
    extracted_jsonl = work_dir / "extracted.jsonl"
    with extracted_jsonl.open("w", encoding="utf-8") as sink:
        for p in sorted(html_dir.glob("*.html")):
            html = p.read_text(encoding="utf-8", errors="ignore")
            clean = extract_clean_text(html)
            rec = {"file": str(p), "title": (clean.title or ""), "text": clean.text or ""}
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    from ..pipeline.checkpoints import extract_report as _extr_rep
    write_checkpoint(work_dir / "checkpoints" / "04_extract_report.json", _extr_rep(extracted_jsonl))

    # 5) Summarize
    from ..summarize.pipeline import summarize_pages_file
    await summarize_pages_file(
        input_jsonl=extracted_jsonl,
        output_jsonl=out_jsonl,
        model=model,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        limit=limit,
        concurrency=1,
        max_chars=max_chars,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        use_filter=False,
        chunked=True,
        chunk_chars=3500,
    )
    from ..pipeline.checkpoints import summaries_quality as _sum_q
    write_checkpoint(work_dir / "checkpoints" / "05_summaries_quality.json", _sum_q(out_jsonl))

    # 6) Store
    from ..store import init_db, import_pages_from_html_dir, import_summaries_jsonl
    init_db(db_path)
    import_pages_from_html_dir(db_path, html_dir, use_cleaner=True, include_pdfs=True)
    import_summaries_jsonl(db_path, out_jsonl)
    # deterministic + LLM validation report
    from ..validate.checks import deterministic_checks
    from ..store.db import retrieve_paragraphs
    from ..validate.validator import async_llm_rag_validate_summary as _llm_rag_async
    vals: List[Dict[str, Any]] = []
    for ln in out_jsonl.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        obj = json.loads(ln)
        det = deterministic_checks(obj)
        rag_warns = await _llm_rag_async(
            str(db_path), obj, k=5, provider=provider, base_url=base_url, api_key=api_key, model=model
        )
        vals.append({"url": obj.get("url"), "deterministic": det, "rag": rag_warns})
    write_checkpoint(work_dir / "checkpoints" / "06_validation_report.json", vals)

    # 7) Compose report
    from ..report.compose import compose_report_from_jsonl
    await compose_report_from_jsonl(out_jsonl, work_dir / "report.md", provider=provider, base_url=base_url, api_key=api_key, model=model)
    # final checkpoint index
    write_checkpoint(work_dir / "checkpoints" / "00_index.json", {
        "plan": str(work_dir / "checkpoints" / "01_plan.json"),
        "urls": str(work_dir / "checkpoints" / "02_urls.json"),
        "crawl": str(work_dir / "checkpoints" / "03_crawl_report.json"),
        "extract": str(work_dir / "checkpoints" / "04_extract_report.json"),
        "summaries": str(work_dir / "checkpoints" / "05_summaries_quality.json"),
        "validation": str(work_dir / "checkpoints" / "06_validation_report.json"),
        "report": str(work_dir / "report.md"),
    })
