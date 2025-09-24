from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Iterable, List, Optional

from pydantic import ValidationError

from ..llm.factory import get_llm_client
from ..filter.relevance import filter_text
from .prompt import build_prompt
from .schema import PageSummary


async def summarize_record(
    oc,
    model: str,
    rec: dict,
    *,
    max_chars: int = 6000,
    options: dict | None = None,
) -> Optional[PageSummary]:
    url = rec.get("url") or rec.get("file") or ""
    title = rec.get("title")
    text = rec.get("text") or ""
    text = text[:max_chars]
    prompt = build_prompt(url=url, title=title, text=text)

    # Start without JSON format for broader model compatibility
    out = await oc.generate(model=model, prompt=prompt, options=options)
    if not out:
        # Retry with JSON format if supported
        out = await oc.generate(model=model, prompt=prompt, format="json", options=options)
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        # Try to extract JSON substring if model returned extra text
        s = out.strip()
        # Strip markdown code fences
        if s.startswith("```"):
            try:
                s = s.strip("`\n ")
                # Remove possible leading 'json' tag
                if s.lower().startswith("json"):
                    s = s[4:].lstrip("\n\r ")
            except Exception:
                pass
        # Try last-to-first braces
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(s[start : end + 1])
            else:
                return None
        except Exception:
            return None
    try:
        return PageSummary.model_validate(data)
    except ValidationError:
        return None


async def summarize_pages_file(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    model: str,
    provider: str = "ollama",
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
    limit: Optional[int] = None,
    concurrency: int = 2,
    max_chars: int = 6000,
    use_filter: bool = False,
    chunked: bool = False,
    chunk_chars: int = 3500,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = 42,
) -> None:
    records: List[dict] = []
    with input_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                continue
            if limit and len(records) >= limit:
                break

    async with get_llm_client(provider, base_url, api_key) as oc:
        sem = asyncio.Semaphore(concurrency)
        out_lines: List[str] = []
        opts = {"temperature": temperature, "top_p": top_p}
        if seed is not None:
            opts["seed"] = seed

        async def run_one(obj: dict) -> None:
            async with sem:
                robj = dict(obj)
                if use_filter and (robj.get("text")):
                    filt, _ = filter_text(robj["text"])  # keep only relevant paras
                    if filt:
                        robj["text"] = filt
                if chunked and len(robj.get("text", "")) > max_chars:
                    summ = await _summarize_chunked(oc, model, robj, max_chars=max_chars, chunk_chars=chunk_chars, options=opts)
                else:
                    summ = await summarize_record(oc, model, robj, max_chars=max_chars, options=opts)
                if summ:
                    out_lines.append(summ.model_dump_json(exclude_none=True))

        await asyncio.gather(*(run_one(o) for o in records))

    with output_jsonl.open("w", encoding="utf-8") as out:
        for line in out_lines:
            out.write(line + "\n")


async def _summarize_chunked(
    oc,
    model: str,
    rec: dict,
    *,
    max_chars: int,
    chunk_chars: int,
    options: dict | None,
):
    text = rec.get("text") or ""
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_chars])
        i += chunk_chars
    parts: List[PageSummary] = []
    for ch in chunks:
        r = dict(rec)
        r["text"] = ch
        part = await summarize_record(oc, model, r, max_chars=max_chars, options=options)
        if part:
            parts.append(part)
    if not parts:
        return None
    return _merge_summaries(parts)


def _merge_summaries(parts: List[PageSummary]) -> PageSummary:
    base = parts[0].model_copy(deep=True)
    def extend_unique(lst, items, key_func=lambda x: x):
        seen = set(key_func(x) for x in lst)
        for it in items:
            k = key_func(it)
            if k not in seen:
                lst.append(it)
                seen.add(k)
    for p in parts[1:]:
        base.key_findings = base.key_findings or []
        extend_unique(base.key_findings, p.key_findings or [], key_func=lambda x: x)
        base.organisms = base.organisms or []
        extend_unique(base.organisms, p.organisms or [], key_func=lambda x: (x.name, x.role))
        base.enzymes = base.enzymes or []
        extend_unique(base.enzymes, p.enzymes or [], key_func=lambda x: (x.name, x.ec_number))
        base.feedstocks = base.feedstocks or []
        extend_unique(base.feedstocks, p.feedstocks or [], key_func=lambda x: x)
        base.starting_substrates = base.starting_substrates or []
        extend_unique(base.starting_substrates, p.starting_substrates or [], key_func=lambda x: x)
        base.metrics = base.metrics or []
        extend_unique(base.metrics, p.metrics or [], key_func=lambda x: (x.kind, x.value, x.unit))
        base.reaction_steps = base.reaction_steps or []
        extend_unique(base.reaction_steps, p.reaction_steps or [], key_func=lambda x: (x.substrate, x.product, getattr(x.enzyme, 'name', None)))
        base.strain_design = base.strain_design or []
        extend_unique(base.strain_design, p.strain_design or [], key_func=lambda x: (x.gene, x.action))
        # Narrative sections: append text with separation if new info appears
        for field in ("summary_long", "methods", "results", "future_perspectives"):
            a = getattr(base, field, None)
            b = getattr(p, field, None)
            if b and (not a or b not in a):
                setattr(base, field, (a + "\n\n" + b) if a else b)
    return base
