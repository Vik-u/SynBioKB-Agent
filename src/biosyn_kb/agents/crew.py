from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .base import Agent, Context, Crew
from ..extract.clean import extract_clean_text
from ..filter.relevance import filter_text
from ..summarize.pipeline import summarize_pages_file
from ..store import init_db, import_pages_from_html_dir, import_summaries_jsonl, query_metrics
from ..entities import extract_entities
from ..validate import normalize_summary_record, validate_summary_record
from ..validate.validator import rag_validate_summary, enrich_summary_xrefs


def build_agents() -> List[Agent]:
    agents: List[Agent] = []

    def extract_agent(ctx: Context) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        extracted_jsonl = ctx.work_dir / "extracted.jsonl"
        with extracted_jsonl.open("w", encoding="utf-8") as sink:
            for p in sorted(ctx.html_dir.glob("*.html")):
                html = p.read_text(encoding="utf-8", errors="ignore")
                clean = extract_clean_text(html, url=None)
                text = clean.text or ""
                if ctx.use_filter:
                    text, _ = filter_text(text)
                rec = {"file": str(p), "title": (clean.title or ""), "text": text}
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ctx.pages_jsonl = extracted_jsonl

    agents.append(
        Agent(
            name="Extractor",
            role="Content Extractor",
            task="Extract clean, relevant body text from saved HTML",
            background="Specialist in boilerplate removal and readability",
            backstory="Trained on scientific pages to capture main content",
            expertise=["HTML parsing", "Boilerplate removal", "Relevance filtering"],
            description="Produces JSONL of {file,title,text} records from HTML files.",
            expected_outputs=["JSONL file with clean text"],
            run_fn=extract_agent,
        )
    )

    def summarize_agent(ctx: Context) -> None:
        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        assert ctx.pages_jsonl, "pages_jsonl missing"
        if ctx.skip_llm:
            # Very lightweight: rule-based summary from the first record
            out_lines: List[str] = []
            with ctx.pages_jsonl.open("r", encoding="utf-8") as fh:
                i = 0
                for line in fh:
                    if ctx.limit and i >= ctx.limit:
                        break
                    i += 1
                    obj = json.loads(line)
                    ents = extract_entities(obj.get("text", ""), seed_chemical=None)
                    summary = {
                        "url": obj.get("file"),
                        "title": obj.get("title"),
                        "chemical": ents.chemical,
                        "organisms": [{"name": o} for o in ents.organisms[:3]],
                        "enzymes": [{"name": f"EC {e}"} for e in ents.enzymes_ec[:3]],
                        "metrics": [{"kind": "yield", "value": y, "unit": "%"} for y in ents.yields_percent[:3]],
                        "key_findings": ["Lightweight heuristic summary"],
                        "evidence": [{"quote": ents.evidence or "", "where": "context"}],
                    }
                    out_lines.append(json.dumps(summary, ensure_ascii=False))
            ctx.summaries_jsonl.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            return

        # LLM path
        # Use chunked + filter conservatively for compute savings if many pages
        use_filter = ctx.use_filter
        chunked = ctx.chunked
        # Summarize (updated API uses provider/base_url/api_key)
        return_ = summarize_pages_file(
            input_jsonl=ctx.pages_jsonl,
            output_jsonl=ctx.summaries_jsonl,
            model=ctx.model,
            provider=getattr(ctx, "provider", "ollama"),
            base_url=getattr(ctx, "base_url", ctx.ollama_url),
            api_key=getattr(ctx, "api_key", None),
            limit=ctx.limit,
            concurrency=1,
            max_chars=ctx.max_chars,
            temperature=ctx.temperature,
            top_p=ctx.top_p,
            seed=ctx.seed,
            use_filter=use_filter,
            chunked=chunked,
            chunk_chars=ctx.chunk_chars,
        )
        # If summarize_pages_file is a coroutine in current version
        if hasattr(return_, "__await__"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(return_)

    agents.append(
        Agent(
            name="Summarizer",
            role="LLM Summarizer",
            task="Create structured + narrative summaries",
            background="Expert in biosynthesis extraction",
            backstory="Works with local Ollama models for cost control",
            expertise=["LLM prompting", "Schema extraction", "Evidence selection"],
            description="Produces PageSummary JSONL from extracted text",
            expected_outputs=["Summaries JSONL"],
            run_fn=summarize_agent,
        )
    )

    def validator_agent(ctx: Context) -> None:
        # Normalize + validate summaries JSONL if present
        if not ctx.summaries_jsonl.exists():
            return
        out_norm = ctx.work_dir / "summaries.normalized.jsonl"
        warnings: List[str] = []
        lines = []
        with ctx.summaries_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj = normalize_summary_record(obj)
                obj = enrich_summary_xrefs(obj)
                warns = validate_summary_record(obj)
                # RAG validation against DB pages
                warns += rag_validate_summary(str(ctx.db_path), obj, k=5)
                # LLM-backed RAG validation for stronger checks
                from ..validate.validator import llm_rag_validate_summary as _llm_rag
                warns += _llm_rag(
                    str(ctx.db_path),
                    obj,
                    k=5,
                    provider=getattr(ctx, "provider", "ollama"),
                    base_url=getattr(ctx, "base_url", ctx.ollama_url),
                    api_key=getattr(ctx, "api_key", None),
                    model=ctx.model,
                )
                warnings.extend(warns)
                lines.append(json.dumps(obj, ensure_ascii=False))
        out_norm.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        # Replace summaries with normalized file
        ctx.summaries_jsonl = out_norm
        ctx.validation_warnings = warnings

    agents.append(
        Agent(
            name="Validator",
            role="Schema Validator & Normalizer",
            task="Normalize units/ECs and validate summary JSON",
            background="Quality control",
            backstory="Keeps the KB clean",
            expertise=["data quality", "validation"],
            description="Writes a normalized JSONL and records warnings",
            expected_outputs=["Normalized summaries JSONL", "Warning list"],
            run_fn=validator_agent,
        )
    )

    def composer_agent(ctx: Context) -> None:
        # Compose a consolidated Markdown report from summaries
        if not ctx.summaries_jsonl.exists():
            return
        from ..report.compose import compose_report_from_jsonl
        report_path = ctx.work_dir / "report.md"
        # Run async function in sync context
        import asyncio
        asyncio.run(
            compose_report_from_jsonl(
                ctx.summaries_jsonl,
                report_path,
                provider=getattr(ctx, "provider", "ollama"),
                base_url=getattr(ctx, "base_url", ctx.ollama_url),
                api_key=getattr(ctx, "api_key", None),
                model=ctx.model,
            )
        )
        # store path on context for later use
        ctx.report_path = report_path

    agents.append(
        Agent(
            name="Composer",
            role="Synthesis & Report Writer",
            task="Compose a consolidated, chronologically organized report across sources",
            background="Scientific synthesis",
            backstory="Combines structured facts into a coherent narrative",
            expertise=["evidence synthesis", "report writing"],
            description="Produces a Markdown report with citations and tables",
            expected_outputs=["report.md"],
            run_fn=composer_agent,
        )
    )

    def experimentalist_agent(ctx: Context) -> None:
        if not ctx.summaries_jsonl.exists():
            return
        from ..report.compose import get_llm_client  # reuse factory via compose import
        import asyncio, json
        prompt = (
            "You are an experimentalist review assistant. From the following study summaries, extract\n"
            "practical experimental insights: host choices, pathway engineering logic, critical enzymes,\n"
            "fermentation modes, and parameters that drove improvements. Provide actionable bullets.\n"
            "Return Markdown.\n"
        )
        lines = [json.loads(ln) for ln in ctx.summaries_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        src = "\n".join(json.dumps({k: v for k, v in obj.items() if k in ("title","year","chemical","organisms","enzymes","metrics","conditions","key_findings","strain_design")}, ensure_ascii=False) for obj in lines)
        async def _run():
            from ..llm.factory import get_llm_client as _get
            async with _get(getattr(ctx, "provider", "ollama"), getattr(ctx, "base_url", ctx.ollama_url), getattr(ctx, "api_key", None)) as oc:
                md = await oc.generate(ctx.model, prompt + "\n" + src, options={"temperature": 0.2, "top_p": 1.0})
            (ctx.work_dir / "experimental_insights.md").write_text(md.strip(), encoding="utf-8")
        import asyncio
        asyncio.run(_run())

    agents.append(
        Agent(
            name="Experimentalist",
            role="Experimental Insight Analyst",
            task="Extract practical experimental insights from studies",
            background="Fermentation & metabolic engineering",
            backstory="Turns results into lab-relevant guidance",
            expertise=["metabolic engineering", "fermentation"],
            description="Writes experimental insights Markdown",
            expected_outputs=["experimental_insights.md"],
            run_fn=experimentalist_agent,
        )
    )

    def compute_agent(ctx: Context) -> None:
        if not ctx.summaries_jsonl.exists():
            return
        import asyncio, json
        prompt = (
            "You are a data analysis assistant. From the study summaries, synthesize metric ranges,\n"
            "identify best-in-class results, normalize units where needed, and discuss trends.\n"
            "Return Markdown with a compact table and commentary.\n"
        )
        lines = [json.loads(ln) for ln in ctx.summaries_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        src = "\n".join(json.dumps({k: v for k, v in obj.items() if k in ("title","year","chemical","metrics","conditions")}, ensure_ascii=False) for obj in lines)
        async def _run():
            from ..llm.factory import get_llm_client as _get
            async with _get(getattr(ctx, "provider", "ollama"), getattr(ctx, "base_url", ctx.ollama_url), getattr(ctx, "api_key", None)) as oc:
                md = await oc.generate(ctx.model, prompt + "\n" + src, options={"temperature": 0.2, "top_p": 1.0})
            (ctx.work_dir / "data_insights.md").write_text(md.strip(), encoding="utf-8")
        import asyncio
        asyncio.run(_run())

    agents.append(
        Agent(
            name="Compute",
            role="Data Trends Analyst",
            task="Synthesize metrics and trends into a compact analysis",
            background="Quantitative analysis",
            backstory="Finds signal in reported numbers",
            expertise=["data analysis", "units"],
            description="Writes data insights Markdown",
            expected_outputs=["data_insights.md"],
            run_fn=compute_agent,
        )
    )

    def store_agent(ctx: Context) -> None:
        init_db(ctx.db_path)
        import_pages_from_html_dir(ctx.db_path, ctx.html_dir, use_cleaner=True, include_pdfs=True)
        if ctx.summaries_jsonl.exists():
            import_summaries_jsonl(ctx.db_path, ctx.summaries_jsonl)

    agents.append(
        Agent(
            name="Store",
            role="Knowledge Base Writer",
            task="Persist pages and summaries into SQLite",
            background="Data engineer for scientific KBs",
            backstory="Ensures durable storage and indexing",
            expertise=["SQLAlchemy", "schema design"],
            description="Imports pages and summaries; creates indices",
            expected_outputs=["SQLite DB with tables filled"],
            run_fn=store_agent,
        )
    )

    def query_agent(ctx: Context) -> None:
        rows = query_metrics(ctx.db_path, limit=5)
        ctx.metrics_preview = rows

    agents.append(
        Agent(
            name="Query",
            role="Metrics Previewer",
            task="List a few metrics for sanity check",
            background="Analyst",
            backstory="Provides a quick glance of results",
            expertise=["analytics"],
            description="Runs a simple metrics query",
            expected_outputs=["JSON list of metrics"],
            run_fn=query_agent,
        )
    )

    return agents


def build_crew() -> Crew:
    agents = build_agents()
    return Crew(
        name="Biosyn Crew",
        purpose="End-to-end biosynthesis info extraction and storage",
        agents=agents,
    )


def run_crew_on_local(html_dir: Path, work_dir: Path, db: Path, out_jsonl: Path, *, limit: int = 1, skip_llm: bool = True) -> Dict[str, Any]:
    crew = build_crew()
    ctx = Context(
        html_dir=html_dir,
        work_dir=work_dir,
        db_path=db,
        summaries_jsonl=out_jsonl,
        limit=limit,
        skip_llm=skip_llm,
        use_filter=True,
        chunked=True,
        chunk_chars=2800,
        max_chars=2000,
    )
    crew.run(ctx)
    return {"status": "ok", "metrics_preview": ctx.metrics_preview}
